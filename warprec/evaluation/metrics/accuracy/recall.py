from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("Recall")
class Recall(UserAverageTopKMetric):
    r"""The Recall@k counts the number of item retrieve correctly,
        over the total number of relevant item in the ground truth.

    The metric formula is defined as:
        Recall@k = (1 / |U_valid|) * sum_{u \in U_valid} (|Rel_u \cap Rec_{u,k}| / |Rel_u|)

    where:
        - $U_{valid}$ is the set of users with at least one relevant item in the ground truth.
        - $Rel_u$ is the set of items relevant to user $u$.
        - $Rec_{u,k}$ is the set of top-k recommended items for user $u$.
        - $| \cdot |$ denotes the cardinality of a set.

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

    Recall@2 = [(1 / 2) + (1 / 2)] / 2 = 0.5

    For further details, please refer to this `link <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

    Args:
        k (int): The cutoff.
        num_users (int): Number of users in the training set.
        *args (Any): The argument list.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
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
