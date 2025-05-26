# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import BaseMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("MAE")
class MAE(BaseMetric):
    """
    Mean Absolute Error (MAE) metric.

    This metric computes the average absolute difference between the predictions and targets.

    The metric formula is defines as:
        MAE = sum(|preds - target|) / total_count

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

    MAE = (|8 - 1| + |7 - 3| + |3 - 2| + |9 - 5|) / 4 = (7 + 4 + 1 + 4) / 4
        = 4

    The normalization happens only for the non-zero elements in the target tensor (the real ratings of the user).

    For further details, please refer to this `link <https://en.wikipedia.org/wiki/Mean_absolute_error>`_.

    Attributes:
        sum_absolute_errors (Tensor): Sum of absolute errors across all batches.
        total_count (Tensor): Total number of elements processed.

    Args:
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    sum_absolute_errors: Tensor
    total_count: Tensor

    def __init__(self, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "sum_absolute_errors", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target = kwargs.get("ground", torch.zeros_like(preds))

        mask = target > 0
        abs_errors = torch.abs(preds[mask] - target[mask])
        self.sum_absolute_errors += abs_errors.sum()
        self.total_count += torch.count_nonzero(target)

    def compute(self):
        """Computes the final metric value."""
        mae = (
            self.sum_absolute_errors / self.total_count
            if self.total_count > 0
            else torch.tensor(0.0)
        )
        return {self.name: mae.item()}

    def reset(self):
        """Reset the metric state."""
        self.sum_absolute_errors.zero_()
        self.total_count.zero_()
