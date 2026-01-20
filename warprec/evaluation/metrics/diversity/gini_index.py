from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("Gini")
class Gini(TopKMetric):
    """The Gini index metric measures the inequality in the distribution of recommended items,
    computed on a per-user basis and averaged over users. This implementation accounts
    for items that were never recommended by applying an offset.

    The metric formula is defines as:
        Gini = (sum_{j=1}^{n_rec} (2*(j + offset) - num_items - 1) * (count_j / free_norm)) / (num_items - 1)

    where:
        - n_rec is the number of items that were recommended at least once,
        - offset = num_items - n_rec (to account for items with zero recommendations),
        - count_j is the recommendation count for the j-th item in ascending order,
        - free_norm is the total number of recommendations made (i.e., sum over users).

    Attributes:
        item_counts (Tensor): Tensor to store the recommendation counts for each item.
        free_norm (Tensor): Total number of recommendations made (accumulated per user).
        num_items (int): Total number of items in the catalog, inferred from the prediction tensor.

    Args:
        k (int): The cutoff for recommendations.
        num_items (int): Number of items in the training set.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {MetricBlock.TOP_K_INDICES}

    item_counts: Tensor
    free_norm: Tensor
    num_items: int

    def __init__(
        self,
        k: int,
        num_items: int,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_items = num_items
        self.add_state(
            "item_counts", default=torch.zeros(self.num_items), dist_reduce_fx="sum"
        )
        # Accumulate the total number of recommendations given (free_norm)
        self.add_state("free_norm", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        # Retrieve top_k_indices from kwargs
        top_k_indices = kwargs.get(f"top_{self.k}_indices")

        # Handle sampled item indices if provided (map local batch indices to global item IDs)
        item_indices = kwargs.get("item_indices")
        if item_indices is not None:
            top_k_indices = torch.gather(item_indices, 1, top_k_indices)

        batch_size = top_k_indices.size(0)
        self.free_norm += torch.tensor(batch_size * self.k, dtype=torch.float)

        # Flatten the indices and update item_counts
        flat_indices = top_k_indices.flatten()

        # Ensure indices are within bounds (safety check)
        flat_indices = flat_indices[flat_indices < self.num_items]

        batch_counts = torch.bincount(flat_indices, minlength=self.num_items)
        self.item_counts += batch_counts.to(self.item_counts)

    def compute(self):
        # Consider only items that have been recommended at least once
        recommended_counts = self.item_counts[self.item_counts > 0].float()

        if (
            recommended_counts.numel() == 0
            or self.num_items == 0
            or self.free_norm == 0
        ):
            return {self.name: torch.tensor(0.0)}

        n_rec_items = recommended_counts.numel()
        sorted_counts, _ = torch.sort(recommended_counts)

        # Offset to account for items never recommended
        offset = self.num_items - n_rec_items
        j = torch.arange(
            n_rec_items, dtype=sorted_counts.dtype, device=sorted_counts.device
        )

        contributions = (2 * (j + offset + 1) - self.num_items - 1) * (
            sorted_counts / self.free_norm
        )

        # Sum contributions and normalize
        gini_index = (torch.sum(contributions) / (self.num_items - 1)).item()
        return {self.name: gini_index}
