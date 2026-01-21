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
        target = kwargs.get("binary_relevance")
        top_k_indices = kwargs.get(f"top_{self.k}_indices")
        item_indices = kwargs.get("item_indices")

        # Handle sampled item indices if provided
        if item_indices is not None:
            top_k_indices = torch.gather(item_indices, 1, top_k_indices)
            rows, cols = target.nonzero(as_tuple=True)
            positive_indices = item_indices[rows, cols]

        else:  # Full evaluation
            _, positive_indices = target.nonzero(as_tuple=True)

        # Accumulate short head and long tail recommendations
        self.short_recs += torch.isin(top_k_indices, self.short_head).sum().float()
        self.long_recs += torch.isin(top_k_indices, self.long_tail).sum().float()
        self.short_gt += torch.isin(positive_indices, self.short_head).sum().float()
        self.long_gt += torch.isin(positive_indices, self.long_tail).sum().float()

    def compute(self):
        # Calculate proportions of hits per group
        pr_short = self.short_recs / (self.short_gt if self.short_gt > 0 else 1.0)
        pr_long = self.long_recs / (self.long_gt if self.long_gt > 0 else 1.0)

        # Handle the case where one group has zero items
        if self.short_gt == 0 or self.long_gt == 0:
            return torch.tensor(0.0)

        pr = torch.stack([pr_short, pr_long])
        # std(unbiased=False) matches numpy std by default for population std
        pop_reo = (torch.std(pr, unbiased=False) / torch.mean(pr)).item()
        return {self.name: pop_reo}
