# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from elliotwo.evaluation.base_metric import BaseMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("AUC")
class AUC(BaseMetric):
    """Computes Area Under the ROC Curve (AUC) using the following approach:

    1. For each user, sorts items by predicted score descending
    2. For each positive item in the sorted list:
       - Calculates its rank position (effective_rank)
       - Counts how many positives precede it (progressive_position)
       - Applies formula: (N - rank + pos_count) / N
       Where N = total_negative_items = (total_items - train_items - target_positives + 1)

    Final AUC is averaged over all positive instances across users.

    Attributes:
        auc (Tensor): The auc value across all users.
        positives (Tensor): The number of positive examples.

    Args:
        train_set (csr_matrix): The training interaction data.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    auc: Tensor
    positives: Tensor

    def __init__(
        self,
        train_set: csr_matrix,
        dist_sync_on_step: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_items = train_set.shape[1]
        self.add_state("auc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("positives", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        target = target.clone()
        target[target > 0] = 1

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

        # AUC compute
        auc_matrix = torch.where(
            sorted_target > 0,
            ((neg_num - effective_rank + progressive_position) / neg_num),
            sorted_target,
        )

        # Update
        self.auc += auc_matrix.sum()
        self.positives += sorted_target.sum()

    def compute(self):
        """Computes the final metric value."""
        return self.auc / self.positives if self.positives > 0 else torch.tensor(0.0)

    def reset(self):
        """Resets the metric state."""
        self.auc.zero_()
        self.positives.zero_()
