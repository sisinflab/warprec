import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import RatingMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("MAE")
class MAE(RatingMetric):
    """Mean Absolute Error (MAE) metric.

    This metric computes the average absolute difference between the predictions and targets.
    """

    def _compute_element_error(self, preds: Tensor, target: Tensor) -> Tensor:
        return torch.abs(preds - target)
