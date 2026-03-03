from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("Recall")
class Recall(UserAverageTopKMetric):
    """The Recall@k counts the number of item retrieve correctly,
    over the total number of relevant item in the ground truth.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        hits = top_k_rel.sum(dim=1).float()
        relevant = target.sum(dim=1).float()

        # Handle cases where there are no relevant items to avoid division by zero
        return torch.where(
            relevant > 0, hits / relevant, torch.tensor(0.0, device=preds.device)
        )
