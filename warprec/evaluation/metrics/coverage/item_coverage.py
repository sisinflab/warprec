# pylint: disable=arguments-differ, unused-argument
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
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

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {MetricBlock.TOP_K_INDICES}

    unique_items: list

    def __init__(
        self, k: int, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("unique_items", default=[], dist_reduce_fx=None)

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )
        self.unique_items.append(top_k_indices.detach().cpu())

    def compute(self):
        """Computes the final metric value."""
        if len(self.unique_items) == 0:
            return torch.tensor(0.0)
        all_items_tensor = torch.cat(self.unique_items, dim=0)
        unique_items: Tensor = torch.unique(all_items_tensor)
        item_coverage = torch.tensor(unique_items.numel())
        return {self.name: item_coverage.item()}

    def reset(self):
        """Resets the metric state."""
        self.unique_items = []
