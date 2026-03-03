from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("LAUC")
class LAUC(UserAverageTopKMetric):
    """Computes Limited Under the ROC Curve (LAUC) using the following approach:

    Args:
        k (int): The cutoff.
        num_users (int): Number of users in the training set.
        num_items (int): Number of items in the training set.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
    }

    def __init__(
        self,
        k: int,
        num_users: int,
        num_items: int,
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k=k, num_users=num_users, **kwargs)
        self.num_items = num_items

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Any, **kwargs: Any
    ) -> Tensor:
        # Compute area and positives of sliced predictions
        area, _ = self.compute_area_stats(preds, target, self.num_items, k=self.k)

        # Normalization by min(positives, k)
        total_positives = target.sum(dim=1)
        normalization = torch.minimum(
            total_positives, torch.tensor(self.k, device=preds.device)
        )

        # LAUC = total_area / min(positives, k)
        return torch.where(
            normalization > 0,
            area / normalization,
            torch.tensor(0.0, device=preds.device),
        )
