# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from warprec.evaluation.base_metric import BaseMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("MSE")
class MSE(BaseMetric):
    """
    Mean Squared Error (MSE) metric.

    This metric computes the average squared difference between the predictions and targets.

    The metric formula is defines as:
        MSE = sum((preds - target)^2) / total_count

    where:
        -preds is the predicted ratings.
        -target are the real ratings of the user.
        -total_count is the total number of elements processed.

    Matrix computation of the metric:
        PREDS                   TARGETS
    +---+---+---+---+       +---+---+---+---+
    | 8 | 2 | 7 | 2 |       | 1 | 0 | 3 | 0 |
    | 5 | 4 | 3 | 9 |       | 0 | 0 | 2 | 5 |
    +---+---+---+---+       +---+---+---+---+

    MSE = ((8 - 1)^2 + (7 - 3)^2 + (3 - 2)^2 + (9 - 5)^2) / 4
        = (49 + 16 + 1 + 16) / 4 = 20.5

    The normalization happens only for the non-zero elements in the target tensor (the real ratings of the user).

    For further details, please refer to this `link <https://en.wikipedia.org/wiki/Mean_squared_error>`_.

    Attributes:
        sum_squared_errors (Tensor): Sum of squared errors across all batches.
        total_count (Tensor): Total number of elements processed.

    Args:
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    sum_squared_errors: Tensor
    total_count: Tensor

    def __init__(self, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "sum_squared_errors", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        mask = target > 0
        squared_errors = (preds[mask] - target[mask]) ** 2
        self.sum_squared_errors += squared_errors.sum()
        self.total_count += torch.count_nonzero(target)

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
