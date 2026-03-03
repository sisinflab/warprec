from torch import Tensor

from warprec.evaluation.metrics.base_metric import RatingMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("MSE")
class MSE(RatingMetric):
    """Mean Squared Error (MSE) metric.

    This metric computes the average squared difference between the predictions and targets.
    """

    def _compute_element_error(self, preds: Tensor, target: Tensor) -> Tensor:
        return (preds - target) ** 2
