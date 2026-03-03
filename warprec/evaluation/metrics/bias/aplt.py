from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("APLT")
class APLT(UserAverageTopKMetric):
    """APLT (Average Proportion of Long-Tail items) is a metric that evaluates
    the proportion of long-tail items present in the top-k recommendations.
    Unlike APLT, which focuses on the number of long-tail recommendations, APLT normalizes
    by the total number of recommended items, providing a proportional measure.

    This metric helps analyze how well a recommendation system balances diversity
    by incorporating less popular items into recommendations while maintaining relevance.

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
        pop_ratio: float = 0.8,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k=k, num_users=num_users, dist_sync_on_step=dist_sync_on_step)
        self.pop_ratio = pop_ratio
        _, lt = self.compute_head_tail(item_interactions, self.pop_ratio)
        self.register_buffer("long_tail", lt)

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        # Retrieve top_k_indices from kwargs
        top_k_indices = kwargs.get(f"top_{self.k}_indices")

        # Remap top_k_indices to global
        item_indices = kwargs.get("item_indices")
        top_k_indices = self.remap_indices(top_k_indices, item_indices)

        # Check which items are in the long tail
        is_long_tail = torch.isin(top_k_indices, self.long_tail)

        # Proportion: Count / k
        return is_long_tail.sum(dim=1).float() / self.k

    @property
    def name(self):
        """The name of the metric."""
        if self.pop_ratio == 0.8:
            return self.__class__.__name__
        return f"APLT[Pop{int(self.pop_ratio * 100)}%]"
