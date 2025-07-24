# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("PopREO")
class PopREO(TopKMetric):
    """Popularity-based Ranking-based Equal Opportunity (PopREO) metric.

    This metric evaluates the fairness of a recommender system by comparing the
    proportion of recommended items from the short head (most popular items) and
    long tail (less popular items) to their respective proportions in the ground truth.
    It calculates the standard deviation of these proportions divided by their mean,
    providing a measure of how equally the system recommends items
    across different popularity groups.

    The metric formula is defined as:
        PopREO = std(pr_short, pr_long) / mean(pr_short, pr_long)

    where:
        -pr_short is the proportion of short head items in the recommendations.
        -pr_long is the proportion of long tail items in the recommendations.

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

    Then we finally extract the short head and long tail items
    from the relevance matrix and ground truth matrix.
    Check BaseMetric for more details on the long tail and short head definition.

    We calculate the proportion of hits as follows:
        - pr_short = sum(short_hits) / sum(short_gt)
        - pr_long = sum(long_hits) / sum(long_gt)

    For further details, please refer to this `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_.

    Attributes:
        short_hits (Tensor): The short head recommendation hits.
        long_hits (Tensor): The long tail recommendation hits.
        short_gt (Tensor): The short head items in the target.
        long_gt (Tensor): The long tail items in the target.

    Args:
        k (int): The cutoff for recommendations.
        item_interactions (Tensor): The counts for item interactions in training set.
        pop_ratio (float): The percentile considered popular.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.TOP_K_INDICES,
        MetricBlock.TOP_K_VALUES,
    }

    short_hits: Tensor
    long_hits: Tensor
    short_gt: Tensor
    long_gt: Tensor

    def __init__(
        self,
        k: int,
        item_interactions: Tensor,
        pop_ratio: float,
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        sh, lt = self.compute_head_tail(item_interactions, pop_ratio)
        self.short_head = sh
        self.long_tail = lt
        self.add_state("short_hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("long_hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("short_gt", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("long_gt", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
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

        # Retrieve user number (consider only user with at least
        # one interaction)
        users = int((target > 0).any(dim=1).sum().item())

        # Expand short head and long tail
        short_head_matrix = self.short_head.expand(
            users, -1
        )  # [batch_size x short_head]
        long_tail_matrix = self.long_tail.expand(users, -1)  # [batch_size x long_tail]

        # Extract short head and long tail items from recommendations
        short_hits = torch.gather(
            rel, 1, short_head_matrix
        )  # [batch_size x short_head]
        long_hits = torch.gather(rel, 1, long_tail_matrix)  # [batch_size x long_tail]

        # Extract short head and long tail items from gt
        short_gt = torch.gather(
            target, 1, short_head_matrix
        )  # [batch_size x short_head]
        long_gt = torch.gather(target, 1, long_tail_matrix)  # [batch_size x long_tail]

        # Update
        self.short_hits += short_hits.sum()
        self.long_hits += long_hits.sum()
        self.short_gt += short_gt.sum()
        self.long_gt += long_gt.sum()

    def compute(self):
        """Computes the final metric value."""
        # Handle division by zero
        if self.short_gt == 0 or self.long_gt == 0:
            return torch.tensor(0.0)

        pr_short = self.short_hits / self.short_gt
        pr_long = self.long_hits / self.long_gt

        # Handle NaN/Inf when both groups have zero probability
        if torch.isnan(pr_short) or torch.isnan(pr_long):
            return torch.tensor(0.0)

        pr = torch.stack([pr_short, pr_long])
        pop_reo = (torch.std(pr, unbiased=False) / torch.mean(pr)).item()
        return {self.name: pop_reo}
