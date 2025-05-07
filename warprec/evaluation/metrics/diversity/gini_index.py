# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("Gini")
class Gini(TopKMetric):
    """The Gini index metric measures the inequality in the distribution of recommended items,
    computed on a per-user basis and averaged over users. This implementation accounts
    for items that were never recommended by applying an offset.

    The metric formula is defines as:
        Gini = 1 - (sum_{j=1}^{n_rec} (2*(j + offset) - num_items - 1) * (count_j / free_norm)) / (num_items - 1)

    where:
        - n_rec is the number of items that were recommended at least once,
        - offset = num_items - n_rec (to account for items with zero recommendations),
        - count_j is the recommendation count for the j-th item in ascending order,
        - free_norm is the total number of recommendations made (i.e., sum over users).

    Attributes:
        recommended_items (list): List of tensors containing recommended item indices.
        free_norm (Tensor): Total number of recommendations made (accumulated per user).
        num_items (int): Total number of items in the catalog, inferred from the prediction tensor.

    Args:
        k (int): The cutoff for recommendations.
        train_set (csr_matrix): The training interaction data.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    recommended_items: list
    free_norm: Tensor
    num_items: int

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_items = train_set.shape[1]
        # Accumulate recommended indices from each update call.
        self.add_state("recommended_items", default=[], dist_reduce_fx=None)
        # Accumulate the total number of recommendations given (free_norm).
        self.add_state("free_norm", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        top_k = torch.topk(preds, self.k, dim=1).indices
        batch_size = top_k.shape[0]
        self.free_norm += torch.tensor(batch_size * self.k, dtype=torch.float)
        self.recommended_items.append(top_k.detach().cpu())

    def compute(self):
        """Computes the final metric value."""
        if (
            len(self.recommended_items) == 0
            or self.num_items is None
            or self.free_norm == 0
        ):
            return torch.tensor(0.0)

        all_items = torch.cat(self.recommended_items, dim=0).view(-1)
        unique, counts = torch.unique(all_items, return_counts=True)
        n_rec_items = unique.numel()
        sorted_counts, _ = torch.sort(counts.float())
        # Offset to account for items never recommended.
        offset = self.num_items - n_rec_items
        j = torch.arange(n_rec_items, dtype=sorted_counts.dtype)
        contributions = (2 * (j + offset + 1) - self.num_items - 1) * (
            sorted_counts / self.free_norm
        )
        # Sum contributions and normalize.
        return 1 - torch.sum(contributions) / (self.num_items - 1)

    def reset(self):
        """Reset the metric state."""
        self.recommended_items = []
        self.free_norm = torch.tensor(0.0)
