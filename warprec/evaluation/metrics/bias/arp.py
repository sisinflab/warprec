from typing import Any, Set

from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("ARP")
class ARP(UserAverageTopKMetric):
    """ARP (Average Recommendation Popularity) is a metric that evaluates
    the average popularity of the top-k recommendations.

    The metric formula is defined as:
        ARP = (1 / |U|) * sum( (1 / k) * sum_{i in L_u} pop(i) )

    where:
        - pop(i) is the popularity of item i (e.g., interaction count).
        - L_u is the set of top-k recommended items for user u.
        - k is the cutoff for recommendations.
        - U is the set of users.

    Matrix computation of the metric:
        PREDS                   POPULARITY TENSOR
    +---+---+---+---+       +---+---+---+---+
    | 8 | 2 | 7 | 2 |       | 10| 5 | 15| 20|
    | 5 | 4 | 3 | 9 |       +---+---+---+---+
    +---+---+---+---+

    1. Extract top-k predictions and get their item indices. Let's assume k=2:
    TOP-K_INDICES
    +---+---+
    | 0 | 2 |
    | 3 | 0 |
    +---+---+

    2. Use these indices to retrieve the popularity from the popularity tensor:
    RECOMMENDED_ITEMS_POP
    +---+---+
    | 10| 15|
    | 20| 10|
    +---+---+

    3. Sum the popularity for each user:
    USER_POP_SUM
    +---+
    | 25|
    | 30|
    +---+

    4. Average over all users and divide by k. For the global metric:
        (25 + 30) / (2 * 2) = 13.75

    For further details, please refer to this `paper <https://arxiv.org/abs/1901.07555>`_.

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
