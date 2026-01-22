from typing import Any, Set

from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("Precision")
class Precision(UserAverageTopKMetric):
    """The Precision@k counts the number of item retrieved correctly,
        over the maximum number of possible retrieve items.

    The metric formula is defined as:
        Precision@k = sum_{u=1}^{n_users} sum_{i=1}^{k} rel_{u,i} / (k * n_users)

    where:
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

    Precision@2 = (1 + 1) / (2 * 2) = 0.5

    For further details, please refer to this `link <https://en.wikipedia.org/wiki/Precision_and_recall>`_.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        return top_k_rel.sum(dim=1).float() / self.k
