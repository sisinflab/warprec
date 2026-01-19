from typing import Any, Set, Tuple

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("GAUC")
class GAUC(UserAverageTopKMetric):
    """Computes Group Area Under the ROC Curve (GAUC) using the following approach:

    Matrix computation of the metric:
        PREDS                   TARGETS
    +---+---+---+---+       +---+---+---+---+
    | 8 | 2 | 7 | 2 |       | 1 | 0 | 1 | 0 |
    | 5 | 4 | 3 | 9 |       | 0 | 0 | 1 | 1 |
    +---+---+---+---+       +---+---+---+---+

    We sort the entire prediction matrix and retrieve the column index:
        SORT PREDS
    +---+---+---+---+
    | 0 | 2 | 1 | 3 |
    | 3 | 0 | 1 | 2 |
    +---+---+---+---+

    then we extract the relevance (original score) for that user in that column:
        SORT REL
    +---+---+---+---+
    | 1 | 1 | 0 | 0 |
    | 1 | 0 | 0 | 1 |
    +---+---+---+---+

    For each user, we compute the negative samples as:
        neg_samples = num_items - train_set - target_set + 1

    the +1 is added to avoid division by zero. The training set
    is extracted from the prediction, which is masked with negative infinite
    in place of the positive samples. The target set is the sum of the
    positive samples for each user.

    We compute the effective extracting the column indices of the sorted relevance:
      EFFECTIVE RANK
    +---+---+---+---+
    | 0 | 1 | 0 | 0 |
    | 0 | 0 | 0 | 3 |
    +---+---+---+---+

    the progressive rank is calculated as the cumulative sum of the sorted relevance:
     PROGRESSIVE RANK
    +---+---+---+---+
    | 1 | 2 | 0 | 0 |
    | 1 | 0 | 0 | 2 |
    +---+---+---+---+

    the AUC scores are computed as follows:
        AUC_{ui} = (neg_samples_{u} - effective_rank_{ui} + progressive_rank_{ui}) / neg_samples_{u}

    The final GAUC is the sum of all AUC normalized by the number of positive samples per user:
        GAUC = sum_{u=1}^{n_users} sum_{i=1}^{items} AUC_{ui} / positives_{u}

    For further details, please refer to this
        `paper <https://www.ijcai.org/Proceedings/2019/0319.pdf>`_.

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
