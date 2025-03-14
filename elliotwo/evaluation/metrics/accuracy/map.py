# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("MAP")
class MAP(TopKMetric):
    """Mean Average Precision (MAP) at K.

    MAP@K calculates the mean of the Average Precision for all users.
    It considers the position of relevant items in the recommendation list.

    Attributes:
        ap_sum (Tensor): The average precision tensor.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The recommendation cutoff.
        dist_sync_on_step (bool): If True, synchronizes the state of the metric across different devices during distributed training.
        *args (Any): Additional arguments to pass to the parent class.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    ap_sum: Tensor
    users: Tensor

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("ap_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the MAP metric state with a batch of predictions."""
        target = target.clone()
        target[target > 0] = 1

        top_k = torch.topk(preds, self.k, dim=1).indices
        rel = torch.gather(target, 1, top_k)

        precision_at_i = rel.cumsum(dim=1) / torch.arange(
            1, self.k + 1, device=rel.device
        )
        normalization = torch.minimum(
            target.sum(dim=1),
            torch.tensor(self.k, dtype=target.dtype, device=target.device),
        )
        ap = (precision_at_i * rel).sum(dim=1) / normalization

        self.ap_sum += ap.sum()
        self.users += target.shape[0]

    def compute(self):
        """Computes the final MAP@K value."""
        return self.ap_sum / self.users if self.users > 0 else torch.tensor(0.0)
