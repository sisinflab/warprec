# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Optional, Dict

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.base_metric import TopKMetric
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
        item_clusters (Tensor): Tensor mapping each item to an item cluster.
        item_counts (Tensor): Tensor of counts of item recommended.
        item_gains (Tensor): Tensor of gains of item recommended.

    Args:
        k (int): Cutoff for top-k recommendations (used by BiasDisparityBR).
        train_set (csr_matrix): Sparse matrix of training interactions (users x items).
        *args (Any): The argument list.
        item_cluster (Optional[Dict[int, int]]): Mapping from item IDs to item cluster IDs.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    item_clusters: Tensor
    item_counts: Tensor
    item_gains: Tensor

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        *args: Any,
        item_cluster: Optional[Dict[int, int]] = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)

        # Handle item clusters
        if item_cluster:
            unique_item_clusters = sorted(set(item_cluster.values()))
            self.n_item_clusters = len(unique_item_clusters)
            item_cluster_remap = {
                cid: idx for idx, cid in enumerate(unique_item_clusters)
            }  # Use appropriate indexes for clusters
            ic = torch.zeros(train_set.shape[1], dtype=torch.long)
            for i, c in item_cluster.items():
                ic[i] = item_cluster_remap[c]
        else:
            self.n_item_clusters = 1
            ic = torch.zeros(train_set.shape[1], dtype=torch.long)
        self.register_buffer("item_clusters", ic)

        # Initialize accumulators for counts and gains
        self.add_state(
            "item_counts",
            torch.zeros(train_set.shape[1], dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "item_gains",
            torch.zeros(train_set.shape[1], dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target = self.discounted_relevance(target)
        _, top_k_indices = torch.topk(preds, self.k, dim=1)

        # Compute counts
        mask_topk = torch.zeros_like(preds, dtype=torch.bool)
        mask_topk.scatter_(1, top_k_indices, True)
        batch_counts = mask_topk.sum(dim=0)

        rel = torch.zeros_like(preds)
        rel.scatter_(1, top_k_indices, target.gather(1, top_k_indices))
        batch_gains = rel.sum(dim=0)

        self.item_counts += batch_counts
        self.item_gains += batch_gains

    def compute(self):
        """Computes the final ItemMADRanking metric value."""
        mask = self.item_counts > 0  # Avoid division by 0

        # Normalized gains per item (only where item was recommended)
        v = torch.zeros_like(self.item_gains)
        v[mask] = self.item_gains[mask] / self.item_counts[mask]

        # Prepare cluster-level accumulators
        sum_per_cluster = torch.zeros(self.n_item_clusters, device=v.device)
        count_per_cluster = torch.zeros(
            self.n_item_clusters, dtype=torch.long, device=v.device
        )

        # Sum of normalized gains per cluster
        sum_per_cluster.scatter_add_(0, self.item_clusters, v)

        # Count of items per cluster (only items that were recommended)
        count_per_cluster.scatter_add_(0, self.item_clusters, mask.long())

        # Avoid division by 0
        mean_per_cluster = torch.zeros_like(sum_per_cluster)
        nonzero_mask = count_per_cluster > 0
        mean_per_cluster[nonzero_mask] = (
            sum_per_cluster[nonzero_mask] / count_per_cluster[nonzero_mask]
        )

        # Compute MAD: mean of absolute differences from the global mean
        global_mean = mean_per_cluster[nonzero_mask].mean()
        mad = (mean_per_cluster[nonzero_mask] - global_mean).abs().mean()

        return {self.name: mad.item()}

    def reset(self):
        """Resets the metric state."""
        self.item_counts.zero_()
        self.item_gains.zero_()
