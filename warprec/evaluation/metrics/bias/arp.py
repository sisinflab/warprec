# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("ARP")
class ARP(TopKMetric):
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
        retrieved_pop (Tensor): The number of interaction for every
            item recommended.
        users (Tensor): The number of users evaluated.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.

    Args:
        k (int): The cutoff for recommendations.
        num_users (int): Number of users in the training set.
        item_interactions (Tensor): The counts for item interactions in training set.
        *args (Any): The argument list.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_INDICES,
    }
    _CAN_COMPUTE_PER_USER: bool = True

    pop: Tensor
    retrieved_pop: Tensor
    users: Tensor
    compute_per_user: bool

    def __init__(
        self,
        k: int,
        num_users: int,
        item_interactions: Tensor,
        *args: Any,
        compute_per_user: bool = False,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.compute_per_user = compute_per_user

        if self.compute_per_user:
            self.add_state(
                "retrieved_pop", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )  # Initialize a tensor to store metric value for each user
        else:
            self.add_state(
                "retrieved_pop", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )  # Initialize a scalar to store global value
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Add popularity counts as buffer
        self.register_buffer("pop", self.compute_popularity(item_interactions))

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        users = kwargs.get("valid_users", self.valid_users(preds))
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )

        # Handle sampled item indices if provided
        item_indices = kwargs.get("item_indices", None)
        if item_indices is not None:
            top_k_indices = torch.gather(item_indices, 1, top_k_indices)

        # Get the popularity of the recommended items
        recommended_items_pop = self.pop[top_k_indices]  # [batch_size x k]

        # Sum the popularity for each user
        user_pop_sum = recommended_items_pop.sum(dim=1).float()  # [batch_size]

        if self.compute_per_user:
            self.retrieved_pop.index_add_(0, user_indices, user_pop_sum)
        else:
            self.retrieved_pop += user_pop_sum.sum()

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final metric value."""
        if self.compute_per_user:
            arp = self.retrieved_pop / self.k
        else:
            arp = (
                self.retrieved_pop / (self.users * self.k)
                if self.users > 0
                else torch.tensor(0.0)
            ).item()
        return {self.name: arp}
