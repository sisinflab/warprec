from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("NumRetrieved")
class NumRetrieved(UserAverageTopKMetric):
    """The NumRetrieved@k counts the number of items retrieved in the top-k list.

    This metric simply counts how many items are present in the recommended list up to
    the specified cutoff k. It does not consider the relevance of the items.

    The metric formula is defined as:
        NumRetrieved@k = (1 / N) * sum_{u=1}^{N} min(k, |L_u|)

    where:
        - N is the total number of users processed across all batches.
        - k is the cutoff.
        - |L_u| is the number of items available in the prediction list for user u
          (in a batch context, this is typically the number of columns in the preds tensor).
          Since the prediction tensor usually has scores for all possible items,
          |L_u| is effectively the total number of items.

    For further details, please refer to the `link <https://github.com/RankSys/RankSys/blob/master/RankSys-metrics/src/main/java/es/uam/eps/ir/ranksys/metrics/basic/NumRetrieved.java>`_
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_VALUES,
    }

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        # Retrieve top_k_values from kwargs
        top_k_values = kwargs.get(f"top_{self.k}_values")

        # Count items that are not -inf (valid recommendations)
        return (~torch.isinf(top_k_values)).sum(dim=1).float()
