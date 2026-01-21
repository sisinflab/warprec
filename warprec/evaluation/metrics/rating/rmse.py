import torch

from warprec.evaluation.metrics.rating import MSE
from warprec.utils.registry import metric_registry


@metric_registry.register("RMSE")
class RMSE(MSE):
    """
    Root Mean Squared Error (RMSE) metric.

    This metric computes the square root of the average squared difference between the predictions and targets.

    The metric formula is defines as:
        RMSE = sqrt(sum((preds - target)^2) / total_count)

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

    RMSE = sqrt(((8 - 1)^2 + (7 - 3)^2 + (3 - 2)^2 + (9 - 5)^2) / 4
         = (49 + 16 + 1 + 16) / 4) = 4.52

    The normalization happens only for the non-zero elements in the target tensor (the real ratings of the user).

    For further details, please refer to this `link <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_.
    """

    def compute(self):
        # Get the MSE per user
        mse = super().compute()[self.name]

        # Apply sqrt to the tensor
        rmse = torch.sqrt(mse)

        return {self.name: rmse}
