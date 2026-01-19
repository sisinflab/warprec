# pylint: disable=arguments-differ, unused-argument, line-too-long, duplicate-code
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("ACLT")
class ACLT(TopKMetric):
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
        long_hits (Tensor): The long tail recommendation hits.
        users (Tensor): The number of users evaluated.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.

    Args:
        k (int): The cutoff for recommendations.
        num_users (int): Number of users in the training set.
        item_interactions (Tensor): The counts for item interactions in training set.
        pop_ratio (float): The percentile considered popular.
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

    long_tail: Tensor
    long_hits: Tensor
    users: Tensor
    compute_per_user: bool

    def __init__(
        self,
        k: int,
        num_users: int,
        item_interactions: Tensor,
        pop_ratio: float,
        *args: Any,
        compute_per_user: bool = False,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.compute_per_user = compute_per_user

        if self.compute_per_user:
            self.add_state(
                "long_hits", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )  # Initialize a tensor to store metric value for each user
            self.add_state(
                "users", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )
        else:
            self.add_state(
                "long_hits", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )  # Initialize a scalar to store global value
            self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Add long tail items as buffer
        _, lt = self.compute_head_tail(item_interactions, pop_ratio)
        self.register_buffer("long_tail", lt)

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users: Tensor = kwargs.get("valid_users", self.valid_users(target))
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )

        # Handle sampled item indices if provided
        item_indices = kwargs.get("item_indices", None)
        if item_indices is not None:
            top_k_indices = torch.gather(item_indices, 1, top_k_indices)

        # Create a boolean tensor where True indicates a long-tail item
        long_tail_mask = torch.isin(top_k_indices, self.long_tail)

        # Sum the number of long-tail hits per user
        long_hits = long_tail_mask.sum(dim=1).float()  # [batch_size]

        if self.compute_per_user:
            self.long_hits.index_add_(0, user_indices, long_hits)

            # Count only users with at least one interaction
            self.users.index_add_(0, user_indices, users)
        else:
            self.long_hits += long_hits.sum()

            # Count only users with at least one interaction
            self.users += users.sum()

    def compute(self):
        """Computes the final metric value."""
        if self.compute_per_user:
            aclt = self.long_hits
            aclt[self.users == 0] = float(
                "nan"
            )  # Set nan for users with no interactions
        else:
            aclt = (
                self.long_hits / self.users if self.users > 0 else torch.tensor(0.0)
            ).item()
        return {self.name: aclt}
