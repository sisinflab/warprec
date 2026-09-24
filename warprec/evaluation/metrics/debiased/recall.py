from typing import Any

import torch
from torch import Tensor

from warprec.evaluation.metrics.debiased.base import InversePropensityMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("IPSRecall")
class IPSRecall(InversePropensityMetric):
    """Recall@k with each hit weighted by the inverse of its propensity.

    The denominator stays the number of relevant items, which is what makes this
    the plain inverse-propensity estimator rather than the self-normalised one.
    """

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        """Compute the corrected recall of each user.

        Args:
            preds (Tensor): The prediction tensor.
            target (Tensor): The relevance of every item.
            top_k_rel (Tensor): The relevance of the top-k items.
            **kwargs (Any): The precomputed blocks.

        Returns:
            Tensor: The metric value per user.
        """
        hits = self.weighted_hits(top_k_rel, self.top_k_indices(preds, **kwargs))
        relevant = target.sum(dim=1).float()

        return torch.where(
            relevant > 0,
            hits.sum(dim=1) / relevant,
            torch.tensor(0.0, device=preds.device),
        )


@metric_registry.register("SNIPSRecall")
class SNIPSRecall(InversePropensityMetric):
    """Recall@k corrected for exposure and normalised by the weights it used.

    Dividing by the summed weights instead of the item count bounds the estimate
    by construction, which is what keeps a single rarely-shown item from carrying
    the whole result.
    """

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        """Compute the self-normalised corrected recall of each user.

        Args:
            preds (Tensor): The prediction tensor.
            target (Tensor): The relevance of every item.
            top_k_rel (Tensor): The relevance of the top-k items.
            **kwargs (Any): The precomputed blocks.

        Returns:
            Tensor: The metric value per user.
        """
        hits = self.weighted_hits(top_k_rel, self.top_k_indices(preds, **kwargs))
        total = self.weight_total(target, kwargs.get("item_indices"))

        return torch.where(
            total > 0,
            hits.sum(dim=1) / total,
            torch.tensor(0.0, device=preds.device),
        )
