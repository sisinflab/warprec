# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from elliotwo.evaluation.base_metric import BaseMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("RMSE")
class RMSE(BaseMetric):
    """
    Root Mean Squared Error (RMSE) metric.

    This metric computes the square root of the average squared difference between the predictions and targets.

    The metric formula is defines as:

    RMSE = sqrt(sum((preds - target)^2) / total_count)

    Matrix computation of the metric:
        PREDS                   TARGETS
    +---+---+---+---+       +---+---+---+---+
    | 8 | 2 | 7 | 2 |       | 1 | 0 | 3 | 0 |
    | 5 | 4 | 3 | 9 |       | 0 | 0 | 2 | 5 |
    +---+---+---+---+       +---+---+---+---+

    RMSE = sqrt(((8 - 1)^2 + (7 - 3)^2 + (3 - 2)^2 + (9 - 5)^2) / 4
         = (49 + 16 + 1 + 16) / 4) = 4.52

    The normalization happens only for the non-zero elements in the target tensor (the real ratings of the user).

    For further details, please refer to this `link <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_.

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
        self.total_count += torch.count_nonzero(target)

    def compute(self):
        """Computes the final metric value."""
        mse = (
            self.sum_squared_errors / self.total_count
            if self.total_count > 0
            else torch.tensor(0.0)
        )
        return torch.sqrt(mse)

    def reset(self):
        """Reset the metric state."""
        self.sum_squared_errors.zero_()
        self.total_count.zero_()
