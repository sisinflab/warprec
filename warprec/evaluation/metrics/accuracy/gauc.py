# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.metrics.base_metric import BaseMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("GAUC")
class GAUC(BaseMetric):
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

    Attributes:
        gauc (Tensor): The gauc value across all users.
        users (Tensor): The number of users.

    Args:
        train_set (csr_matrix): The training interaction data.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    gauc: Tensor
    users: Tensor

    def __init__(
        self,
        train_set: csr_matrix,
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_items = train_set.shape[1]
        self.add_state("gauc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target = kwargs.get("binary_relevance", torch.zeros_like(preds))

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

        # Effective rank
        col_indices = torch.arange(sorted_target.shape[1]).repeat(
            sorted_target.shape[0], 1
        )
        effective_rank = torch.where(
            sorted_target == 1, col_indices, torch.tensor(0.0)
        )  # [batch_size x items]

        # Progressive position
        cumsum = torch.cumsum(sorted_target, dim=1)
        progressive_position = torch.where(
            sorted_target == 1, cumsum - 1, sorted_target
        )  # [batch_size x items]

        # GAUC compute
        auc_matrix = torch.where(
            sorted_target > 0,
            ((neg_num - effective_rank + progressive_position) / neg_num),
            sorted_target,
        )
        positive_sum = sorted_target.sum(dim=1)  # [batch_size]
        auc_tensor = auc_matrix.sum(dim=1)  # [batch_size]
        gauc_values = torch.where(
            positive_sum != 0, auc_tensor / positive_sum, positive_sum
        )

        # Update
        self.gauc += gauc_values.sum()

        # Count only users with at least one interaction
        self.users += (target > 0).any(dim=1).sum().item()

    def compute(self):
        """Computes the final metric value."""
        score = self.gauc / self.users if self.users > 0 else torch.tensor(0.0)
        return {self.name: score.item()}

    def reset(self):
        """Resets the metric state."""
        self.gauc.zero_()
        self.users.zero_()
