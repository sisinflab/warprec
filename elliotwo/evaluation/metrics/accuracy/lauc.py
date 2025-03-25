# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("LAUC")
class LAUC(TopKMetric):
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

    For further details, please refer to this `paper <https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf>`_.

    Attributes:
        lauc (Tensor): The lauc value across all users.
        users (Tensor): The number of users.

    Args:
        k (int): The cutoff.
        train_set (csr_matrix): The training interaction data.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    lauc: Tensor
    users: Tensor

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        dist_sync_on_step: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_items = train_set.shape[1]
        self.add_state("lauc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        target = self.binary_relevance(target)

        # Negative samples
        train_set = torch.isinf(preds).logical_and(preds < 0).sum(dim=1)  # [batch_size]
        target_set = target.sum(dim=1)  # [batch_size]
        neg_num: Tensor = self.num_items - train_set - target_set + 1  # [batch_size]
        neg_num = neg_num.unsqueeze(1)

        # Sorted recommendations
        _, sorted_preds = torch.sort(
            preds, dim=1, descending=True
        )  # [batch_size x items]
        sorted_target = torch.gather(target, 1, sorted_preds)
        sorted_target = sorted_target[:, : self.k]  # [batch_size x top_k]

        # Effective rank
        col_indices = torch.arange(sorted_target.shape[1]).repeat(
            sorted_target.shape[0], 1
        )
        effective_rank = torch.where(
            sorted_target == 1, col_indices, torch.tensor(0.0)
        )  # [batch_size x top_k]

        # Progressive position
        cumsum = torch.cumsum(sorted_target, dim=1)
        progressive_position = torch.where(
            sorted_target == 1, cumsum - 1, sorted_target
        )  # [batch_size x top_k]

        # LAUC compute
        auc_matrix = torch.where(
            sorted_target > 0,
            ((neg_num - effective_rank + progressive_position) / neg_num),
            sorted_target,
        )
        normalization = torch.minimum(
            target.sum(dim=1), torch.tensor(self.k)
        )  # [batch_size]
        auc_tensor = auc_matrix.sum(dim=1)  # [batch_size]
        lauc_values = auc_tensor / normalization

        # Update
        self.lauc += lauc_values.sum()
        self.users += target.shape[0]

    def compute(self):
        """Computes the final metric value."""
        return self.lauc / self.users if self.users > 0 else torch.tensor(0.0)

    def reset(self):
        """Resets the metric state."""
        self.lauc.zero_()
        self.users.zero_()
