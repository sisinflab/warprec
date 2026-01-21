from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("ACLT")
class ACLT(UserAverageTopKMetric):
    """ACLT (Average Coverage of Long-Tail items) is a metric that evaluates the
    extent to which a recommendation system provides recommendations from the long-tail
    of item popularity. The long-tail is determined based on a given popularity percentile threshold.

    This metric is designed to assess recommendation diversity by measuring the
    proportion of recommended long-tail items relative to all recommendations. A higher
    ACLT value indicates a system that effectively recommends less popular items.

    The metric formula is defined as:
        ACLT = sum(long_hits) / users

    where:
        -long_hits are the number of recommendation in the long tail.

    Matrix computation of the metric:
        PREDS                   TARGETS
    +---+---+---+---+       +---+---+---+---+
    | 8 | 2 | 7 | 2 |       | 1 | 0 | 1 | 0 |
    | 5 | 4 | 3 | 9 |       | 0 | 0 | 1 | 1 |
    +---+---+---+---+       +---+---+---+---+

    We extract the top-k predictions and get their column index. Let's assume k=2:
      TOP-K
    +---+---+
    | 0 | 2 |
    | 3 | 0 |
    +---+---+

    then we extract the relevance (original score) for that user in that column but maintaining the original dimensions:
           REL
    +---+---+---+---+
    | 0 | 0 | 1 | 0 |
    | 0 | 0 | 0 | 1 |
    +---+---+---+---+

    Then we finally extract the long tail items from the relevance matrix.
    Check BaseMetric for more details on the long tail definition.

    For further details, please refer to this `paper <https://arxiv.org/abs/1901.07555>`_.

    Attributes:
        long_tail (Tensor): The lookup tensor of long tail items.

    Args:
        k (int): The cutoff for recommendations.
        num_users (int): Number of users in the training set.
        item_interactions (Tensor): The counts for item interactions in training set.
        pop_ratio (float): The percentile considered popular.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_INDICES,
    }

    long_tail: Tensor

    def __init__(
        self,
        k: int,
        num_users: int,
        item_interactions: Tensor,
        pop_ratio: float,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            k=k, num_users=num_users, dist_sync_on_step=dist_sync_on_step, **kwargs
        )
        _, lt = self.compute_head_tail(item_interactions, pop_ratio)
        self.register_buffer("long_tail", lt)

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        # Retrieve top_k_indices from kwargs
        top_k_indices = kwargs.get(f"top_{self.k}_indices")

        # Handle sampled item indices if provided (map local batch indices to global item IDs)
        item_indices = kwargs.get("item_indices")
        if item_indices is not None:
            top_k_indices = torch.gather(item_indices, 1, top_k_indices)

        # Check which items are in the long tail
        is_long_tail = torch.isin(top_k_indices, self.long_tail)

        # Sum hits per user (Count of Long Tail items)
        return is_long_tail.sum(dim=1).float()
