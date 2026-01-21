import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import RatingMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("MAE")
class MAE(RatingMetric):
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
    """

    def _compute_element_error(self, preds: Tensor, target: Tensor) -> Tensor:
        return torch.abs(preds - target)
