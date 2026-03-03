import torch

from warprec.evaluation.metrics.rating import MSE
from warprec.utils.registry import metric_registry


@metric_registry.register("RMSE")
class RMSE(MSE):
    """Root Mean Squared Error (RMSE) metric.

    This metric computes the square root of the average squared difference between the predictions and targets.
    """

    def compute(self):
        # Get the MSE per user
        mse = super().compute()[self.name]

        # Apply sqrt to the tensor
        rmse = torch.sqrt(mse)

        return {self.name: rmse}
