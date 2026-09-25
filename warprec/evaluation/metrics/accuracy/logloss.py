from typing import Any, Set

import torch
from torch import Tensor
from torch.nn import functional as F

from warprec.evaluation.metrics.base_metric import RatingMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("LogLoss")
class LogLoss(RatingMetric):
    """Binary cross-entropy between the scores and the relevance labels.

    Every ranking metric in this framework judges an ordering, which says nothing
    about whether a model's scores are calibrated: a model can rank perfectly while
    being confidently wrong about how likely any single interaction is. LogLoss is
    the counterpart, and it is the measure the click-through-rate literature
    reports, so it is what makes the context-aware models here comparable with the
    numbers published for them.

    It reads the scores as **logits** and squashes them with a sigmoid, because
    nothing in the framework constrains a model's output to [0, 1]. A model that
    already emits a probability is therefore not the one to read this on.

    Items the run did not score are skipped rather than counted as confident
    negatives: the seen-item mask sets them to negative infinity, and letting those
    through would add a term of zero loss for most of the catalogue and drive the
    number towards zero for reasons that have nothing to do with the model.

    It pairs naturally with the `sampled` strategy, where each user contributes one
    positive and a fixed number of negatives. Under `full` ranking the labels are
    overwhelmingly negative, and the value reflects that imbalance.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {MetricBlock.BINARY_RELEVANCE}

    def _compute_element_error(self, preds: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(preds, target, reduction="none")

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Accumulate the per-user loss over every item the run actually scored.

        Args:
            preds (Tensor): The predicted scores, read as logits.
            user_indices (Tensor): The users the rows belong to.
            **kwargs (Any): The precomputed evaluation blocks.

        Raises:
            ValueError: If the binary relevance block was not supplied.
        """
        relevance = kwargs.get("binary_relevance")
        if relevance is None:
            raise ValueError(
                "LogLoss needs the binary relevance of every scored item, which "
                "the evaluator did not provide."
            )

        scored = torch.isfinite(preds)

        # The masked entries must not reach the loss at all: an infinite logit
        # gives an infinite term that no later masking can take back out.
        logits = torch.where(scored, preds, torch.zeros_like(preds))
        losses = self._compute_element_error(logits, relevance.to(logits.dtype))
        losses = torch.where(scored, losses, torch.zeros_like(losses))

        self.error_sum.index_add_(
            0, user_indices, losses.sum(dim=1).to(self.error_sum.dtype)
        )
        self.total_count.index_add_(
            0, user_indices, scored.sum(dim=1).to(self.total_count.dtype)
        )
