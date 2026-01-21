from typing import Any, Set

import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("ItemMADRanking")
class ItemMADRanking(TopKMetric):
    """Item MAD Ranking (ItemMADRanking) metric.

    This metric measures the disparity in item exposure across different item clusters
    in the top-k recommendations, by computing the Mean Absolute Deviation (MAD) of the average
    discounted relevance scores per cluster. The goal is to evaluate whether some item clusters
    receive consistently higher or lower exposure than others.

    Formally, the metric is defined as:

        MAD = mean_c(|mean_gain(c) - mean_global|)

    where:
        - mean_gain(c) is the average discounted gain of items in cluster c (only for recommended items),
        - mean_global is the average of mean_gain(c) over all item clusters with at least one recommended item.

    This metric is useful to detect disparities in ranking quality across clusters (e.g., genres, popularity buckets),
    independent of the absolute relevance of items.

    The metric uses a discounted relevance model (e.g., log-based) applied to the top-k predictions,
    and tracks the average relevance score each item receives when recommended. These per-item scores
    are then aggregated by cluster to compute the cluster-level mean gains and their deviation.

    For further details, please refer to this `link <https://link.springer.com/article/10.1007/s11257-020-09285-1>`_.

    Attributes:
        num_items (int): Number of items in the training set.
        item_clusters (Tensor): Tensor mapping each item to an item cluster.
        item_counts (Tensor): Tensor of counts of item recommended.
        item_gains (Tensor): Tensor of gains of item recommended.

    Args:
        k (int): Cutoff for top-k recommendations.
        num_items (int): Number of items in the training set.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.DISCOUNTED_RELEVANCE,
        MetricBlock.TOP_K_DISCOUNTED_RELEVANCE,
        MetricBlock.TOP_K_INDICES,
    }

    num_items: int
    item_clusters: Tensor
    item_counts: Tensor
    item_gains: Tensor

    def __init__(
        self,
        k: int,
        num_items: int,
        item_cluster: Tensor,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k=k, dist_sync_on_step=dist_sync_on_step)
        self.num_items = num_items

        # Register item clusters as buffer
        self.register_buffer("item_clusters", item_cluster)
        self.n_item_clusters = int(item_cluster.max().item()) + 1

        # Initialize accumulators
        self.add_state("item_counts", torch.zeros(num_items), dist_reduce_fx="sum")
        self.add_state("item_gains", torch.zeros(num_items), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        top_k_gains = kwargs.get(f"top_{self.k}_discounted_relevance")
        top_k_indices = kwargs.get(f"top_{self.k}_indices")

        # Handle sampled item indices if provided (map local batch indices to global item IDs)
        item_indices = kwargs.get("item_indices")
        if item_indices is not None:
            top_k_indices = torch.gather(item_indices, 1, top_k_indices)

        # Flatten for accumulation
        flat_indices = top_k_indices.flatten()
        flat_gains = top_k_gains.flatten().float()

        # Accumulate counts (1 for every appearance in top-k)
        self.item_counts.index_add_(
            0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float)
        )

        # Accumulate gains (discounted relevance from GT)
        self.item_gains.index_add_(0, flat_indices, flat_gains)

    def compute(self):
        # Compute average gain per item (only for recommended items)
        recommended_mask = self.item_counts > 0

        if not recommended_mask.any():
            return {self.name: torch.tensor(0.0)}

        item_avg_gains = torch.zeros_like(self.item_gains)
        item_avg_gains[recommended_mask] = (
            self.item_gains[recommended_mask] / self.item_counts[recommended_mask]
        )

        # Aggregate per cluster
        # Get clusters for recommended items
        rec_indices = torch.where(recommended_mask)[0]
        rec_clusters = self.item_clusters[rec_indices]
        rec_gains = item_avg_gains[rec_indices]

        # Sum gains and counts per cluster
        cluster_sum_gains = torch.zeros(self.n_item_clusters, device=self.device)
        cluster_counts = torch.zeros(self.n_item_clusters, device=self.device)

        cluster_sum_gains.index_add_(0, rec_clusters, rec_gains)
        cluster_counts.index_add_(0, rec_clusters, torch.ones_like(rec_gains))

        # Compute Mean per cluster
        valid_clusters = cluster_counts > 0

        if not valid_clusters.any():
            return {self.name: torch.tensor(0.0)}

        cluster_means = (
            cluster_sum_gains[valid_clusters] / cluster_counts[valid_clusters]
        )

        # Compute MAD (Pairwise differences)
        if cluster_means.numel() < 2:
            mad = torch.tensor(0.0, device=self.device)
        else:
            # Vectorized pairwise absolute difference
            diffs = (cluster_means.unsqueeze(0) - cluster_means.unsqueeze(1)).abs()
            # Sum upper triangle
            pairwise_sum = diffs.triu(diagonal=1).sum()
            num_pairs = cluster_means.numel() * (cluster_means.numel() - 1) / 2
            mad = pairwise_sum / num_pairs

        return {self.name: mad.item()}
