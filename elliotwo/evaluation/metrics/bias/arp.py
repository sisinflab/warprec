# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("ARP")
class ARP(TopKMetric):
    """

    Attributes:
        total_pop (Tensor): The total number of interaction for every
            item recommended.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff for recommendations.
        train_set (csr_matrix): The training interaction data.
        dist_sync_on_step (bool): Torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    total_pop: Tensor
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
        self.pop = self.compute_popularity(train_set).unsqueeze(0)
        self.add_state("total_pop", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        target = target.clone()
        target[target > 0] = 1

        top_k_values, top_k_indices = torch.topk(preds, self.k, dim=1)
        rel = torch.zeros_like(preds)
        rel.scatter_(
            dim=1, index=top_k_indices, src=top_k_values
        )  # [batch_size x items]
        rel = rel * target  # [batch_size x items]
        rel[rel > 0] = 1

        batch_pop = self.pop.repeat(target.shape[0], 1)  # [batch_size x items]

        self.total_pop += (rel * batch_pop).sum()
        self.users += target.shape[0]

    def compute(self):
        """Computes the final metric value."""
        return self.total_pop / (self.users * self.k)

    def reset(self):
        """Resets the metric state."""
        self.total_pop.zero_()
        self.users.zero_()
