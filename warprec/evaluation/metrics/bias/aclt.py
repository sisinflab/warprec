# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("ACLT")
class ACLT(TopKMetric):
    """ACLT (Average Coverage of Long-Tail items) is a metric that evaluates the
    extent to which a recommendation system provides recommendations from the long-tail
    of item popularity. The long-tail is determined based on a given popularity percentile threshold.

    This metric is designed to assess recommendation diversity by measuring the
    proportion of recommended long-tail items relative to all recommendations. A higher
    ACLT value indicates a system that effectively recommends less popular items.

    The metric formula is defined as:
        ACLT = sum(long_hits) / users

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

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_INDICES,
        MetricBlock.TOP_K_VALUES,
    }

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

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_values: Tensor = kwargs.get(
            f"top_{self.k}_values", self.top_k_values_indices(preds, self.k)[0]
        )
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )

        rel = torch.zeros_like(preds)
        rel.scatter_(
            dim=1, index=top_k_indices, src=top_k_values
        )  # [batch_size x items]
        rel = rel * target  # [batch_size x items]
        rel[rel > 0] = 1

        # Expand long tail
        long_tail_matrix = self.long_tail.expand(
            int(users), -1
        )  # [batch_size x long_tail]

        # Extract long tail items from recommendations
        long_hits = torch.gather(rel, 1, long_tail_matrix)  # [batch_size x long_tail]

        # Update
        self.long_hits += long_hits.sum()
        self.users += users

    def compute(self):
        """Computes the final metric value."""
        aclt = self.long_hits / self.users if self.users > 0 else torch.tensor(0.0)
        return {self.name: aclt.item()}

    def reset(self):
        """Resets the metric state."""
        self.long_hits.zero_()
        self.users.zero_()
