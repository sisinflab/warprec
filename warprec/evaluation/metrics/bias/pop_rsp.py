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

    Attributes:
        short_head (Tensor): The lookup tensor of short head items.
        long_tail (Tensor): The lookup tensor of long tail items.
        total_short (Tensor): The total number of short head items.
        total_long (Tensor): The total number of long tail items.
        short_recs (Tensor): The short head recommendations.
        long_recs (Tensor): The long tail recommendations.
        avail_short (Tensor): Short head slots that were open to be recommended.
        avail_long (Tensor): Long tail slots that were open to be recommended.

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
    avail_short: Tensor
    avail_long: Tensor

    def __init__(
        self,
        k: int,
        item_interactions: Tensor,
        pop_ratio: float = 0.8,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.pop_ratio = pop_ratio
        self.add_state("short_recs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("long_recs", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Add short head and long tail items as buffer
        sh, lt = self.compute_head_tail(item_interactions, self.pop_ratio)
        self.register_buffer("short_head", sh)
        self.register_buffer("long_tail", lt)

        self.register_buffer("total_short", torch.tensor(len(sh), dtype=torch.float))
        self.register_buffer("total_long", torch.tensor(len(lt), dtype=torch.float))

        self.add_state("avail_short", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("avail_long", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        top_k_indices = kwargs.get(f"top_{self.k}_indices")
        item_indices = kwargs.get("item_indices")

        # Remap top_k_indices to global
        top_k_indices = self.remap_indices(top_k_indices, item_indices)

        # Accumulate short head and long tail recommendations
        self.short_recs += torch.isin(top_k_indices, self.short_head).sum().float()
        self.long_recs += torch.isin(top_k_indices, self.long_tail).sum().float()

        # A group's rate is how often it was recommended out of how often it
        # could have been, so the denominator counts what was on offer to each
        # user rather than the size of the group. The evaluator has already put
        # everything unavailable beyond reach — items the user saw in training,
        # and anything outside a restricted candidate set — so what remains
        # finite is exactly what could have been recommended.
        offered = torch.isfinite(preds)
        if item_indices is None:
            columns = torch.arange(preds.size(1), device=preds.device).expand_as(preds)
        else:
            columns = item_indices

        self.avail_short += (offered & torch.isin(columns, self.short_head)).sum()
        self.avail_long += (offered & torch.isin(columns, self.long_tail)).sum()

    def compute(self):
        """Computes the final metric value."""
        # Handle division by zero
        if self.avail_short == 0 or self.avail_long == 0:
            return {self.name: torch.tensor(0.0)}

        pr_short = self.short_recs / self.avail_short
        pr_long = self.long_recs / self.avail_long
        pr = torch.stack([pr_short, pr_long])

        # Handle the case where mean is zero
        if torch.mean(pr) == 0:
            return {self.name: torch.tensor(0.0)}

        pop_rsp = torch.std(pr, unbiased=False) / torch.mean(pr)
        return {self.name: pop_rsp.item()}

    @property
    def name(self):
        """The name of the metric."""
        if self.pop_ratio == 0.8:
            return self.__class__.__name__
        return f"PopRSP[Pop{int(self.pop_ratio * 100)}%]"
