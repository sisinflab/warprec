# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("ItemCoverage")
class ItemCoverage(TopKMetric):
    """The ItemCoverage@k metric counts the number of unique items
       that were recommended across all users.

    Attributes:
        unique_items (list): The list of unique items per batch.

    Args:
        k (int): The cutoff.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    unique_items: list

    def __init__(
        self, k: int, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("unique_items", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, target: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        top_k = torch.topk(preds, self.k, dim=1).indices
        self.unique_items.append(top_k.detach().cpu())

    def compute(self):
        """Computes the final metric value."""
        if len(self.unique_items) == 0:
            return torch.tensor(0.0)
        all_items_tensor = torch.cat(self.unique_items, dim=0)
        unique_items: Tensor = torch.unique(all_items_tensor)
        return torch.tensor(unique_items.numel())

    def reset(self):
        """Resets the metric state."""
        self.unique_items = []
