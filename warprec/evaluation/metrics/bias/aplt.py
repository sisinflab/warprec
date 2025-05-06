# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("APLT")
class APLT(TopKMetric):
    """APLT (Average Proportion of Long-Tail items) is a metric that evaluates
    the proportion of long-tail items present in the top-k recommendations.
    Unlike APLT, which focuses on the number of long-tail recommendations, APLT normalizes
    by the total number of recommended items, providing a proportional measure.

    This metric helps analyze how well a recommendation system balances diversity
    by incorporating less popular items into recommendations while maintaining relevance.

    The metric formula is defined as:
        APLT = sum(long_hits) / (users * k)

    where:
        -long_hits are the number of recommendation in the long tail.

    Matrix computation of the metric:
        PREDS                   TARGETS
    +---+---+---+---+       +---+---+---+---+
    | 8 | 2 | 7 | 2 |       | 1 | 0 | 1 | 0 |
    | 5 | 4 | 3 | 9 |       | 0 | 0 | 1 | 1 |
    +---+---+---+---+       +---+---+---+---+

    We extract the top-k predictions and get their column index. Let's assume k=2:
      TOP-K
    +---+---+
    | 0 | 2 |
    | 3 | 0 |
    +---+---+

    then we extract the relevance (original score) for that user in that column but maintaining the original dimensions:
           REL
    +---+---+---+---+
    | 0 | 0 | 1 | 0 |
    | 0 | 0 | 0 | 1 |
    +---+---+---+---+

    Then we finally extract the long tail items from the relevance matrix.
    Check BaseMetric for more details on the long tail definition.

    For further details, please refer to this `paper <https://arxiv.org/abs/1901.07555>`_.

    Attributes:
        long_hits (Tensor): The long tail recommendation hits.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff for recommendations.
        train_set (csr_matrix): The training interaction data.
        pop_ratio (float): The percentile considered popular.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    long_hits: Tensor
    users: Tensor

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        pop_ratio: float,
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        _, lt = self.compute_head_tail(train_set, pop_ratio)
        self.long_tail = lt
        self.add_state("long_hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        target = self.binary_relevance(target)
        top_k_values, top_k_indices = torch.topk(preds, self.k, dim=1)
        rel = torch.zeros_like(preds)
        rel.scatter_(
            dim=1, index=top_k_indices, src=top_k_values
        )  # [batch_size x items]
        rel = rel * target  # [batch_size x items]
        rel[rel > 0] = 1

        # Retrieve user number (consider only user with at least
        # one interaction)
        self.users += (target > 0).any(dim=1).sum().item()

        # Expand long tail
        long_tail_matrix = self.long_tail.expand(
            int(self.users), -1
        )  # [batch_size x long_tail]

        # Extract long tail items from recommendations
        long_hits = torch.gather(rel, 1, long_tail_matrix)  # [batch_size x long_tail]

        # Update
        self.long_hits += long_hits.sum()

    def compute(self):
        """Computes the final metric value."""
        return self.long_hits / (self.users * self.k)

    def reset(self):
        """Resets the metric state."""
        self.long_hits.zero_()
        self.users.zero_()
