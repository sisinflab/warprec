# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("APLT")
class APLT(TopKMetric):
    """APLT (Average Proportion of Long-Tail items) is a metric that evaluates
    the proportion of long-tail items present in the top-k recommendations.
    Unlike ACLT, which focuses on the number of long-tail recommendations, APLT normalizes
    by the total number of recommended items, providing a proportional measure.

    This metric helps analyze how well a recommendation system balances diversity
    by incorporating less popular items into recommendations while maintaining relevance.

    Attributes:
        long_hits (Tensor): The long tail recommendation hits.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff for recommendations.
        train_set (csr_matrix): The training interaction data.
        pop_ratio (float): The percentile considered popular.
        dist_sync_on_step (bool): Torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    long_hits: Tensor
    users: Tensor

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        pop_ratio: float,
        dist_sync_on_step: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        _, lt = self.compute_popularity(train_set, pop_ratio)
        self.long_tail = lt
        self.add_state("long_hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
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

        # Retrieve user number (batch_size)
        users = target.shape[0]

        # Expand long tail
        long_tail_matrix = self.long_tail.expand(users, -1)  # [batch_size x long_tail]

        # Extract long tail items from recommendations
        long_hits = torch.gather(rel, 1, long_tail_matrix)  # [batch_size x long_tail]

        # Update
        self.long_hits += long_hits.sum()
        self.users += users

    def compute(self):
        """Computes the final metric value."""
        return self.long_hits / (self.users * self.k)

    def reset(self):
        """Resets the metric state."""
        self.long_hits.zero_()
        self.users.zero_()
