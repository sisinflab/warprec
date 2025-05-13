# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("NumRetrieved")
class NumRetrieved(TopKMetric):
    """The NumRetrieved@k counts the number of items retrieved in the top-k list.

    This metric simply counts how many items are present in the recommended list up to
    the specified cutoff k. It does not consider the relevance of the items.

    The metric formula is defined as:
        NumRetrieved@k = (1 / N) * sum_{u=1}^{N} min(k, |L_u|)

    where:
        - N is the total number of users processed across all batches.
        - k is the cutoff.
        - |L_u| is the number of items available in the prediction list for user u
          (in a batch context, this is typically the number of columns in the preds tensor).
          Since the prediction tensor usually has scores for all possible items,
          |L_u| is effectively the total number of items.

    Attributes:
        cumulative_count (Tensor): Sum of retrieved items per user.
        users (Tensor): Number of user with at least 1 relevant item.

    Args:
        k (int): The cutoff.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    cumulative_count: Tensor
    users: Tensor

    def __init__(
        self, k: int, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any
    ):
        super().__init__(k, *args, dist_sync_on_step=dist_sync_on_step, **kwargs)

        self.add_state(
            "cumulative_count", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        users = (target > 0).any(dim=1).sum().item()
        num_items = preds.size(1)

        self.cumulative_count += users * min(self.k, num_items)
        self.users += users

    def compute(self):
        """Computes the final metric value."""
        num_retrieved = (
            self.cumulative_count / self.users if self.users > 0 else torch.tensor(0.0)
        )

        return {self.name: num_retrieved.item()}

    def reset(self):
        """Resets the metric state."""
        self.cumulative_count.zero_()
        self.users.zero_()
