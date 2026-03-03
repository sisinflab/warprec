from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("MAR")
class MAR(UserAverageTopKMetric):
    """Mean Average Recall (MAR) at K."""

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        recall_at_i = top_k_rel.cumsum(dim=1) / target.sum(dim=1).unsqueeze(1).clamp(
            min=1
        )  # [batch_size, k]
        normalization = torch.minimum(
            target.sum(dim=1),
            torch.tensor(self.k, dtype=target.dtype, device=target.device),
        )  # [batch_size]

        # Compute AR per user
        return torch.where(
            normalization > 0,
            (recall_at_i * top_k_rel).sum(dim=1) / normalization,
            torch.tensor(0.0, device=self._device),
        )  # [batch_size]
