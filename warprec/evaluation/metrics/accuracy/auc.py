from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import BaseMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("AUC")
class AUC(BaseMetric):
    """Computes Area Under the ROC Curve (AUC) using the following approach:

    Matrix computation of the metric:
        PREDS                   TARGETS
    +---+---+---+---+       +---+---+---+---+
    | 8 | 2 | 7 | 2 |       | 1 | 0 | 1 | 0 |
    | 5 | 4 | 3 | 9 |       | 0 | 0 | 1 | 1 |
    +---+---+---+---+       +---+---+---+---+

    We sort the entire prediction matrix and retrieve the column index:
        SORT PREDS
    +---+---+---+---+
    | 0 | 2 | 1 | 3 |
    | 3 | 0 | 1 | 2 |
    +---+---+---+---+

    then we extract the relevance (original score) for that user in that column:
        SORT REL
    +---+---+---+---+
    | 1 | 1 | 0 | 0 |
    | 1 | 0 | 0 | 1 |
    +---+---+---+---+

    For each user, we compute the negative samples as:
        neg_samples = num_items - train_set - target_set + 1

    the +1 is added to avoid division by zero. The training set
    is extracted from the prediction, which is masked with negative infinite
    in place of the positive samples. The target set is the sum of the
    positive samples for each user.

    We compute the effective extracting the column indices of the sorted relevance:
      EFFECTIVE RANK
    +---+---+---+---+
    | 0 | 1 | 0 | 0 |
    | 0 | 0 | 0 | 3 |
    +---+---+---+---+

    the progressive rank is calculated as the cumulative sum of the sorted relevance:
     PROGRESSIVE RANK
    +---+---+---+---+
    | 1 | 2 | 0 | 0 |
    | 1 | 0 | 0 | 2 |
    +---+---+---+---+

    the AUC scores are computed as follows:
        AUC_{ui} = (neg_samples_{u} - effective_rank_{ui} + progressive_rank_{ui}) / neg_samples_{u}

    The final AUC is the sum of all AUC scores divided by the number of positive samples:
        AUC = sum_{u=1}^{n_users} sum_{i=1}^{items} AUC_{ui} / positives

    For further details,please refer
        to this `link <https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve>`_.

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
