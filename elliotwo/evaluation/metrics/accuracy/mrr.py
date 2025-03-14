# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("MRR")
class MRR(TopKMetric):
    """Mean Reciprocal Rank (MRR) at K.

    MRR measures the position of the first relevant item in the recommendation list.

    Attributes:
        reciprocal_rank_sum (Tensor): The reciprocal rank sum tensor.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The recommendation cutoff.
        dist_sync_on_step (bool): If True, synchronizes the state of the metric across different devices during distributed training.
        *args (Any): Additional arguments to pass to the parent class.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    reciprocal_rank_sum: Tensor
    users: Tensor

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state(
            "reciprocal_rank_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the MRR metric state with a batch of predictions."""
        target = target.clone()
        target[target > 0] = 1

        top_k = torch.topk(preds, self.k, dim=1).indices
        rel = torch.gather(target, 1, top_k)

        # Find the first relevant item's rank
        reciprocal_ranks = (rel.argmax(dim=1) + 1).float().reciprocal()
        reciprocal_ranks[rel.sum(dim=1) == 0] = 0  # Assign 0 if no relevant items

        self.reciprocal_rank_sum += reciprocal_ranks.sum()
        self.users += target.shape[0]

    def compute(self):
        """Computes the final MRR@K value."""
        return (
            self.reciprocal_rank_sum / self.users
            if self.users > 0
            else torch.tensor(0.0)
        )
