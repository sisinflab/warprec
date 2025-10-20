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
        - pr_short = sum(short_recs) / sum(short_gt)
        - pr_long = sum(long_recs) / sum(long_gt)

    For further details, please refer to this `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_.

    Attributes:
        short_head (Tensor): The lookup tensor of short head items.
        long_tail (Tensor): The lookup tensor of long tail items.
        short_recs (Tensor): The short head recommendations.
        long_recs (Tensor): The long tail recommendations.
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
    }

    short_head: Tensor
    long_tail: Tensor
    short_recs: Tensor
    long_recs: Tensor
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
        self.add_state("short_recs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("long_recs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("short_gt", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("long_gt", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Add short head and long tail items as buffer
        sh, lt = self.compute_head_tail(item_interactions, pop_ratio)
        self.register_buffer("short_head", sh)
        self.register_buffer("long_tail", lt)

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )

        # Handle sampled item indices if provided
        item_indices = kwargs.get("item_indices", None)
        if item_indices is not None:
            # Map top_k_indices from local batch indices to global item indices
            top_k_indices_global = torch.gather(item_indices, 1, top_k_indices)

            # Find the global item IDs of the positive interactions in the target matrix
            positive_indices_global = item_indices[target.nonzero(as_tuple=True)]

            # Count recommended items from each group
            short_recs = torch.isin(top_k_indices_global, self.short_head).sum().float()
            long_recs = torch.isin(top_k_indices_global, self.long_tail).sum().float()

            # Count ground truth items from each group
            short_gt = (
                torch.isin(positive_indices_global, self.short_head).sum().float()
            )
            long_gt = torch.isin(positive_indices_global, self.long_tail).sum().float()

        else:  # Full evaluation
            # Count recommended items from each group
            short_recs = torch.isin(top_k_indices, self.short_head).sum().float()
            long_recs = torch.isin(top_k_indices, self.long_tail).sum().float()

            # Get item IDs of positive interactions in the full target matrix
            positive_indices = target.nonzero(as_tuple=True)[1]

            # Count ground truth items from each group
            short_gt = torch.isin(positive_indices, self.short_head).sum().float()
            long_gt = torch.isin(positive_indices, self.long_tail).sum().float()

        # Update
        self.short_recs += short_recs
        self.long_recs += long_recs
        self.short_gt += short_gt
        self.long_gt += long_gt

    def compute(self):
        """Computes the final metric value."""
        # Calculate proportions of hits per group
        pr_short = self.short_recs / (self.short_gt if self.short_gt > 0 else 1.0)
        pr_long = self.long_recs / (self.long_gt if self.long_gt > 0 else 1.0)

        # Handle the case where one group has zero items
        if self.short_gt == 0 or self.long_gt == 0:
            return torch.tensor(0.0)

        pr = torch.stack([pr_short, pr_long])
        pop_reo = (torch.std(pr, unbiased=False) / torch.mean(pr)).item()
        return {self.name: pop_reo}
