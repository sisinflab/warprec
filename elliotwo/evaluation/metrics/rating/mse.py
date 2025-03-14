# pylint: disable=arguments-differ
from typing import Any

import torch
from torch import Tensor
from elliotwo.evaluation.base_metric import BaseMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("MSE")
class MSE(BaseMetric):
    """
    Mean Squared Error (MSE) metric.

    This metric computes the average squared difference between the predictions and targets.

    Attributes:
        sum_squared_errors (Tensor): Sum of squared errors across all batches.
        total_count (Tensor): Total number of elements processed.

    Args:
        dist_sync_on_step (bool): Torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    sum_squared_errors: Tensor
    total_count: Tensor

    def __init__(self, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "sum_squared_errors", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        mask = target > 0
        squared_errors = (preds[mask] - target[mask]) ** 2
        self.sum_squared_errors += squared_errors.sum()
        self.total_count += target.numel()

    def compute(self):
        """Computes the final metric value."""
        return (
            self.sum_squared_errors / self.total_count
            if self.total_count > 0
            else torch.tensor(0.0)
        )

    def reset(self):
        """Reset the metric state."""
        self.sum_squared_errors.zero_()
        self.total_count.zero_()
