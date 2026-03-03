from typing import Any, Set

from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("ARP")
class ARP(UserAverageTopKMetric):
    """ARP (Average Recommendation Popularity) is a metric that evaluates
    the average popularity of the top-k recommendations.

    Attributes:
        pop (Tensor): The lookup tensor of item popularity.

    Args:
        k (int): The cutoff for recommendations.
        num_users (int): Number of users in the training set.
        item_interactions (Tensor): The counts for item interactions in training set.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_INDICES,
    }

    pop: Tensor

    def __init__(
        self,
        k: int,
        num_users: int,
        item_interactions: Tensor,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k=k, num_users=num_users, dist_sync_on_step=dist_sync_on_step)
        self.register_buffer("pop", self.compute_popularity(item_interactions))

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        # Retrieve top_k_indices from kwargs
        top_k_indices = kwargs.get(f"top_{self.k}_indices")

        # Remap top_k_indices to global
        item_indices = kwargs.get("item_indices")
        top_k_indices = self.remap_indices(top_k_indices, item_indices)

        # Retrieve popularity for the recommended items
        # Shape: [batch_size, k]
        recommended_items_pop = self.pop[top_k_indices]

        # Average popularity per user
        return recommended_items_pop.mean(dim=1).float()
