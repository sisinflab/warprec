from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import BaseMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("AUC")
class AUC(BaseMetric):
    """Computes Area Under the ROC Curve (AUC)

    Attributes:
        total_area (Tensor): The accumulated area under the curve.
        total_positives (Tensor): The accumulated number of positive samples.

    Args:
        num_items (int): Number of items in the training set.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
    }

    total_area: Tensor
    total_positives: Tensor

    def __init__(
        self,
        num_items: int,
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_items = num_items
        self.add_state("total_area", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "total_positives", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, **kwargs: Any):
        target = kwargs.get("binary_relevance", torch.zeros_like(preds))

        # Compute area and positives
        area, positives = self.compute_area_stats(preds, target, self.num_items)

        # Accumulate
        self.total_area += area.sum()
        self.total_positives += positives.sum()

    def compute(self):
        score = (
            self.total_area / self.total_positives
            if self.total_positives > 0
            else torch.tensor(0.0)
        )
        return {self.name: score.item()}
