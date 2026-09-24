from typing import Any

import torch
from torch import Tensor

from warprec.evaluation.metrics.debiased.base import InversePropensityMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("IPSDCG")
class IPSDCG(InversePropensityMetric):
    """DCG@k with each gain weighted by the inverse of its propensity.

    The gain is left undiscounted by an ideal ranking on purpose. nDCG divides by
    an ideal DCG that does not depend on the propensities, so correcting only the
    numerator would produce a number that is neither bounded by one nor the
    estimator the literature defines. The unnormalised form is what stays
    unbiased, which is why this metric is not called 'IPSnDCG'.
    """

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        """Compute the corrected discounted gain of each user.

        Args:
            preds (Tensor): The prediction tensor.
            target (Tensor): The relevance of every item.
            top_k_rel (Tensor): The relevance of the top-k items.
            **kwargs (Any): The precomputed blocks.

        Returns:
            Tensor: The metric value per user.
        """
        hits = self.weighted_hits(top_k_rel, self.top_k_indices(preds, **kwargs))
        return self.dcg(hits)


@metric_registry.register("SNIPSDCG")
class SNIPSDCG(InversePropensityMetric):
    """DCG@k corrected for exposure and normalised by the weights it used."""

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        """Compute the self-normalised corrected discounted gain of each user.

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
            self.dcg(hits) / total,
            torch.tensor(0.0, device=preds.device),
        )
