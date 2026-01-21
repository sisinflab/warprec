from torch import Tensor

from warprec.evaluation.metrics.base_metric import RatingMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("MSE")
class MSE(RatingMetric):
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
    """

    def _compute_element_error(self, preds: Tensor, target: Tensor) -> Tensor:
        return (preds - target) ** 2
