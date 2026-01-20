from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("PopRSP")
class PopRSP(TopKMetric):
    """Popularity-based Ranking-based Statistical Parity (PopRSP) metric.

    This metric evaluates the disparity in recommendation performance
    between popular (short head) and less popular (long tail) items.
    It calculates the standard deviation of precision across these
    two groups, normalized by their mean, to assess the balance in
    recommendation exposure.

    The metric formula is defined as:
        PopRSP = std(pr_short, pr_long) / mean(pr_short, pr_long)

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
    from the relevance matrix and training set matrix.
    Check BaseMetric for more details on the long tail and short head definition.

    We calculate the proportion of hits as follows:
        - pr_short = sum(short_hits) / sum(short_gt)
        - pr_long = sum(long_hits) / sum(long_gt)

    For further details, please refer to this `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_.

    Attributes:
        short_head (Tensor): The lookup tensor of short head items.
        long_tail (Tensor): The lookup tensor of long tail items.
        total_short (Tensor): The total number of short head items.
        total_long (Tensor): The total number of long tail items.
        short_recs (Tensor): The short head recommendations.
        long_recs (Tensor): The long tail recommendations.

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
    total_short: Tensor
    total_long: Tensor
    short_recs: Tensor
    long_recs: Tensor

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

        # Add short head and long tail items as buffer
        sh, lt = self.compute_head_tail(item_interactions, pop_ratio)
        self.register_buffer("short_head", sh)
        self.register_buffer("long_tail", lt)

        # Store the total number of items in each group
        self.register_buffer("total_short", torch.tensor(len(sh), dtype=torch.float))
        self.register_buffer("total_long", torch.tensor(len(lt), dtype=torch.float))

    def update(self, preds: Tensor, **kwargs: Any):
        top_k_indices = kwargs.get(f"top_{self.k}_indices")
        item_indices = kwargs.get("item_indices")

        # Handle sampled item indices if provided
        item_indices = kwargs.get("item_indices", None)
        if item_indices is not None:
            top_k_indices = torch.gather(item_indices, 1, top_k_indices)

        # Accumulate short head and long tail recommendations
        self.short_recs += torch.isin(top_k_indices, self.short_head).sum().float()
        self.long_recs += torch.isin(top_k_indices, self.long_tail).sum().float()

    def compute(self):
        """Computes the final metric value."""
        # Handle division by zero
        if self.total_short == 0 or self.total_long == 0:
            return {self.name: torch.tensor(0.0)}

        pr_short = self.short_recs / self.total_short
        pr_long = self.long_recs / self.total_long
        pr = torch.stack([pr_short, pr_long])

        # Handle the case where mean is zero
        if torch.mean(pr) == 0:
            return {self.name: torch.tensor(0.0)}

        pop_rsp = torch.std(pr, unbiased=False) / torch.mean(pr)
        return {self.name: pop_rsp.item()}
