from typing import Any, Set

from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("MRR")
class MRR(UserAverageTopKMetric):
    """Mean Reciprocal Rank (MRR) at K.

    MRR measures the position of the first relevant item in the recommendation list.

    The metric formula is defined as:
        MRR@K = sum_{u=1}^{n_users} (1 / rank_u) / n_users

    where:
        - rank_u is the position of the first relevant item in the recommendation list.

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

    The reciprocal rank is calculated as the inverse of the rank of the first relevant item:
    RECIPROCAL RANK
    +----+---+
    | .5 | 1 |
    +----+---+

    MRR@2 = (0.5 + 1) / 2 = 0.75
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        # Find the first relevant item's rank
        reciprocal_ranks = (top_k_rel.argmax(dim=1) + 1).float().reciprocal()
        reciprocal_ranks[top_k_rel.sum(dim=1) == 0] = 0  # Assign 0 if no relevant items

        return reciprocal_ranks
