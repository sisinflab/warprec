# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("Recall")
class Recall(TopKMetric):
    r"""The Recall@k counts the number of item retrieve correctly,
        over the total number of relevant item in the ground truth.

    The metric formula is defined as:
        Recall@k = (1 / |U_valid|) * sum_{u \in U_valid} (|Rel_u \cap Rec_{u,k}| / |Rel_u|)

    where:
        - $U_{valid}$ is the set of users with at least one relevant item in the ground truth.
        - $Rel_u$ is the set of items relevant to user $u$.
        - $Rec_{u,k}$ is the set of top-k recommended items for user $u$.
        - $| \cdot |$ denotes the cardinality of a set.

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

    Recall@2 = [(1 / 2) + (1 / 2)] / 2 = 0.5

    For further details, please refer to this `link <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

    Attributes:
        retrieved (Tensor): The number of relevant items retrieved.
        users (Tensor): The number of users evaluated.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.

    Args:
        k (int): The cutoff.
        num_users (int): Number of users in the training set.
        *args (Any): The argument list.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }
    _CAN_COMPUTE_PER_USER: bool = True

    retrieved: Tensor
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
                "retrieved", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )  # Initialize a tensor to store metric value for each user
        else:
            self.add_state(
                "retrieved", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )  # Initialize a scalar to store global value
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel: Tensor = kwargs.get(
            f"top_{self.k}_binary_relevance",
            self.top_k_relevance(preds, target, self.k),
        )

        hits = top_k_rel.sum(dim=1).float()
        relevant = target.sum(dim=1).float()

        if self.compute_per_user:
            self.retrieved.index_add_(
                0,
                user_indices,
                torch.where(relevant > 0, hits / relevant, torch.tensor(0.0)),
            )  # Index metric values per user
        else:
            self.retrieved += (
                torch.where(relevant > 0, hits / relevant, torch.tensor(0.0))
                .sum()
                .float()
            )  # Count global 'retrieved' items

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final metric value."""
        if self.compute_per_user:
            return {self.name: self.retrieved}  # Return the tensor with per_user metric
        else:
            recall = (
                self.retrieved / self.users if self.users > 0 else torch.tensor(0.0)
            )
            return {self.name: recall.item()}  # Return the metric value
