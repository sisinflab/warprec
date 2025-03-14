# pylint: disable=arguments-differ
from typing import Any

import torch
from torch import Tensor
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("UserCoverage")
class UserCoverage(TopKMetric):
    """The UserCoverage@k metric counts the number of users
       that received at least one recommendation.

    Attributes:
        covered_users (Tensor): The number of users with at least one recommendation.

    Args:
        k (int): The cutoff.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    covered_users: Tensor

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("covered_users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        top_k = torch.topk(preds, self.k, dim=1).indices
        self.covered_users += top_k.shape[0]

    def compute(self):
        """Computes the final metric value."""
        return self.covered_users

    def reset(self):
        """Resets the metric state."""
        self.covered_users.zero_()
