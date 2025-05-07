# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("UserCoverage")
class UserCoverage(TopKMetric):
    """The UserCoverage@k metric counts the number of users
       that received at least one recommendation.

    Attributes:
        user_coverage (Tensor): The number of users with at least one recommendation.

    Args:
        k (int): The cutoff.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    user_coverage: Tensor

    def __init__(
        self, k: int, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("user_coverage", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        top_k = torch.topk(preds, self.k, dim=1).indices
        self.user_coverage += top_k.shape[0]

    def compute(self):
        """Computes the final metric value."""
        return {self.name: self.user_coverage.item()}

    def reset(self):
        """Resets the metric state."""
        self.user_coverage.zero_()
