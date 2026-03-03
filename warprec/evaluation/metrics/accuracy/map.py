from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("MAP")
class MAP(UserAverageTopKMetric):
    """Mean Average Precision (MAP) at K."""

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        precision_at_i = top_k_rel.cumsum(dim=1) / torch.arange(
            1, self.k + 1, device=top_k_rel.device
        )  # [batch_size, k]
        normalization = torch.minimum(
            target.sum(dim=1),
            torch.tensor(self.k, dtype=target.dtype, device=target.device),
        )  # [batch_size]

        # Compute AP per user
        return torch.where(
            normalization > 0,
            (precision_at_i * top_k_rel).sum(dim=1) / normalization,
            torch.tensor(0.0, device=self._device),
        )  # [batch_size]
