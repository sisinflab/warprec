# pylint: disable=arguments-differ, unused-argument, line-too-long, duplicate-code
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("NumRetrieved")
class NumRetrieved(TopKMetric):
    """The NumRetrieved@k counts the number of items retrieved in the top-k list.

    This metric simply counts how many items are present in the recommended list up to
    the specified cutoff k. It does not consider the relevance of the items.

    The metric formula is defined as:
        NumRetrieved@k = (1 / N) * sum_{u=1}^{N} min(k, |L_u|)

    where:
        - N is the total number of users processed across all batches.
        - k is the cutoff.
        - |L_u| is the number of items available in the prediction list for user u
          (in a batch context, this is typically the number of columns in the preds tensor).
          Since the prediction tensor usually has scores for all possible items,
          |L_u| is effectively the total number of items.

    For further details, please refer to the `link <https://github.com/RankSys/RankSys/blob/master/RankSys-metrics/src/main/java/es/uam/eps/ir/ranksys/metrics/basic/NumRetrieved.java>`_

    Attributes:
        num_retrieved (Tensor): Counts of retrieved items per user.
        users (Tensor): Number of user with at least 1 relevant item.
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
        MetricBlock.TOP_K_VALUES,
    }
    _CAN_COMPUTE_PER_USER: bool = True

    num_retrieved: Tensor
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
                "num_retrieved", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )  # Initialize a tensor to store metric value for each user
        else:
            self.add_state(
                "num_retrieved", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )  # Initialize a scalar to store global value
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_values: Tensor = kwargs.get(
            f"top_{self.k}_values", self.top_k_values_indices(preds, self.k)[0]
        )

        if self.compute_per_user:
            self.num_retrieved.index_add_(
                0, user_indices, (~torch.isinf(top_k_values)).sum(dim=1).float()
            )
        else:
            self.num_retrieved += (~torch.isinf(top_k_values)).sum().float()

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final metric value."""
        if self.compute_per_user:
            num_retrieved = self.num_retrieved
        else:
            num_retrieved = int(
                (
                    self.num_retrieved / self.users
                    if self.users > 0
                    else torch.tensor(0.0)
                ).item()
            )
        return {self.name: num_retrieved}
