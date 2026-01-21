from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("MAR")
class MAR(UserAverageTopKMetric):
    """Mean Average Recall (MAR) at K.

    MAR@K calculates the mean of the Average Recall for all users.

    The metric formula is defined as:
        MAR@K = sum_{u=1}^{n_users} sum_{i=1}^{k} (R@i * rel_{u,i}) / n_users

    where:
        - R@i is the recall at i-th position.
        - rel_{u,i} is the relevance of the i-th item for user u.

    Matrix computation of the metric:
        PREDS                   TARGETS
    +---+---+---+---+       +---+---+---+---+
    | 8 | 2 | 7 | 2 |       | 1 | 0 | 1 | 0 |
    | 5 | 4 | 3 | 9 |       | 0 | 0 | 1 | 1 |
    +---+---+---+---+       +---+---+---+---+

    We extract the top-k predictions and get their column index. Let's assume k=2:
      TOP-K
    +---+---+
    | 0 | 2 |
    | 3 | 0 |
    +---+---+

    then we extract the relevance (original score) for that user in that column:
       REL
    +---+---+
    | 0 | 1 |
    | 1 | 0 |
    +---+---+

    The recall at i-th position is calculated as the sum of the relevant items
    divided by the number of relevant items:
       RECALL
    +----+----+
    | 0  | .5 |
    | .5 | 0  |
    +----+----+

    the normalization is the minimum between the number of relevant items and k:
    NORMALIZATION
    +---+---+
    | 2 | 2 |
    +---+---+

    MAR@2 = 0.5 / 2 + 0.5 / 2 = 0.5

    For further details, please refer to this `link <https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#So-Why-Did-I-Bother-Defining-Recall?>`_.
    """

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
