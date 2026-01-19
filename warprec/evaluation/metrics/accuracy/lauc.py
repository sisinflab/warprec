from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("LAUC")
class LAUC(UserAverageTopKMetric):
    """Computes Limited Under the ROC Curve (LAUC) using the following approach:

    The metric formula is defined as:
        AUC = (sum_{i=1}^{M} rank_i - ((M x (M + 1)) / 2)) / (M x N)

    where:
        -M is the number of positive samples.
        -N is the number of negative samples.
        -rank_i is the rank of the i-th positive sample.

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

    The final LAUC is the sum of all AUC normalized by the number of positive samples limited to k:
        LAUC = sum_{u=1}^{n_users} sum_{i=1}^{items} AUC_{ui} / positives@k_{u}

    For further details, please refer to this
        `paper <https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf>`_.

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
