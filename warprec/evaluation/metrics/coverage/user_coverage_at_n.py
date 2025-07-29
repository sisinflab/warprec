# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("UserCoverageAtN")
class UserCoverageAtN(TopKMetric):
    """The UserCoverageAtN counts the number of user that retrieved
        correctly at least N recommendations.

    This metric measures the system's ability to provide a sufficiently long
    recommendation list for users.

    The metric formula is defined as:
        UserCoverageAtN = sum_{u=1}^{N_total} [|L_u| >= k]

    where:
        - N_total is the total number of users processed across all batches.
        - k is the cutoff.
        - |L_u| is the number of items available in the prediction list for user u.
        - [|L_u| >= k] is either 1 or 0.

    Attributes:
        users (Tensor): Number of user with at least 1 relevant item.

    Args:
        k (int): The cutoff.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    users: Tensor

    def __init__(
        self, k: int, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)

        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        self.users += (preds > 0).sum(dim=1).ge(self.k).sum()

    def compute(self):
        """Computes the final metric value."""
        user_coverage_at_n = int(self.users.item())
        return {self.name: user_coverage_at_n}
