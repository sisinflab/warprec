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
        item_counts (Tensor): The tensor of item counts.

    Args:
        k (int): The cutoff.
        num_items (int): Number of items in the training set.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {MetricBlock.TOP_K_INDICES}

    item_counts: Tensor

    def __init__(
        self,
        k: int,
        num_items: int,
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state(
            "item_counts", default=torch.zeros(num_items), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )

        # Handle sampled item indices if provided
        item_indices = kwargs.get("item_indices", None)
        if item_indices is not None:
            top_k_indices = torch.gather(kwargs.get("item_indices"), 1, top_k_indices)

        # Flatten the indices
        flat_indices = top_k_indices.flatten()

        batch_counts = torch.bincount(flat_indices, minlength=len(self.item_counts))
        self.item_counts += batch_counts.to(self.item_counts)

    def compute(self):
        """Computes the final metric value."""
        item_coverage = (self.item_counts > 0).sum().item()
        return {self.name: item_coverage}
