from typing import Any, Set, Tuple

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("GAUC")
class GAUC(UserAverageTopKMetric):
    """Computes Group Area Under the ROC Curve (GAUC) using the following approach:

    Args:
        num_items (int): Number of items in the training set.
        num_users (int): Number of users in the training set.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
    }

    def __init__(
        self,
        num_items: int,
        num_users: int,
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k=0, num_users=num_users, dist_sync_on_step=dist_sync_on_step)
        self.num_items = num_items

    def unpack_inputs(self, preds: Tensor, **kwargs: Any) -> Tuple[Tensor, Tensor, Any]:
        target = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        return target, users, None

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Any, **kwargs: Any
    ) -> Tensor:
        # Compute area and positives per user
        area, positives = self.compute_area_stats(preds, target, self.num_items, k=None)

        # GAUC = total_area / total_positives
        return torch.where(
            positives > 0, area / positives, torch.tensor(0.0, device=preds.device)
        )
