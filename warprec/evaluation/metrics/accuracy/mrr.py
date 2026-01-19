# pylint: disable=arguments-differ, unused-argument, line-too-long, duplicate-code
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("MRR")
class MRR(TopKMetric):
    """Mean Reciprocal Rank (MRR) at K.

    MRR measures the position of the first relevant item in the recommendation list.

    The metric formula is defined as:
        MRR@K = sum_{u=1}^{n_users} (1 / rank_u) / n_users

    where:
        - rank_u is the position of the first relevant item in the recommendation list.

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

    then we extract the relevance (original score) for that user in that column:
       REL
    +---+---+
    | 0 | 1 |
    | 1 | 0 |
    +---+---+

    The reciprocal rank is calculated as the inverse of the rank of the first relevant item:
    RECIPROCAL RANK
    +----+---+
    | .5 | 1 |
    +----+---+

    MRR@2 = (0.5 + 1) / 2 = 0.75

    Attributes:
        reciprocal_rank (Tensor): The reciprocal rank tensor.
        users (Tensor): The number of users evaluated.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.

    Args:
        k (int): The recommendation cutoff.
        num_users (int): Number of users in the training set.
        *args (Any): Additional arguments to pass to the parent class.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }
    _CAN_COMPUTE_PER_USER: bool = True

    reciprocal_rank: Tensor
    users: Tensor
    compute_per_user: bool

    def __init__(
        self,
        k: int,
        num_users: int,
        *args: Any,
        compute_per_user: bool = False,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.compute_per_user = compute_per_user

        if self.compute_per_user:
            self.add_state(
                "reciprocal_rank", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )  # Initialize a tensor to store metric value for each user
            self.add_state(
                "users", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )
        else:
            self.add_state(
                "reciprocal_rank", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )  # Initialize a scalar to store global value
            self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Updates the MRR metric state with a batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users: Tensor = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel: Tensor = kwargs.get(
            f"top_{self.k}_binary_relevance",
            self.top_k_relevance(preds, target, self.k),
        )

        # Find the first relevant item's rank
        reciprocal_ranks = (top_k_rel.argmax(dim=1) + 1).float().reciprocal()
        reciprocal_ranks[top_k_rel.sum(dim=1) == 0] = 0  # Assign 0 if no relevant items

        if self.compute_per_user:
            self.reciprocal_rank.index_add_(
                0, user_indices, reciprocal_ranks
            )  # Index metric values per user

            # Count only users with at least one interaction
            self.users.index_add_(0, user_indices, users)
        else:
            self.reciprocal_rank += reciprocal_ranks.sum()  # Sum the total RR values

            # Count only users with at least one interaction
            self.users += users.sum()

    def compute(self):
        """Computes the final MRR@K value."""
        if self.compute_per_user:
            mrr = self.reciprocal_rank  # Return the tensor with per_user metric
            mrr[self.users == 0] = float(
                "nan"
            )  # Set nan for users with no interactions
        else:
            mrr = (
                self.reciprocal_rank / self.users
                if self.users > 0
                else torch.tensor(0.0)
            ).item()  # Return the metric value
        return {self.name: mrr}
