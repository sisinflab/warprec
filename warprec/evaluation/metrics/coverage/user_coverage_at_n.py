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
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    users: Tensor

    def __init__(self, k: int, dist_sync_on_step: bool = False, **kwargs: Any):
        super().__init__(k, dist_sync_on_step)

        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        # Count how many items have a score > 0 for each user
        valid_items_per_user = (preds > 0).sum(dim=1)

        # Check if the count is >= k
        satisfied_users = valid_items_per_user.ge(self.k).sum()

        self.users += satisfied_users

    def compute(self):
        return {self.name: int(self.users.item())}
