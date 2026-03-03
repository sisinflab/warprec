from typing import Any, Set

from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("MRR")
class MRR(UserAverageTopKMetric):
    """Mean Reciprocal Rank (MRR) at K. MRR measures the position of the first
    relevant item in the recommendation list."""

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        # Find the first relevant item's rank
        reciprocal_ranks = (top_k_rel.argmax(dim=1) + 1).float().reciprocal()
        reciprocal_ranks[top_k_rel.sum(dim=1) == 0] = 0  # Assign 0 if no relevant items

        return reciprocal_ranks
