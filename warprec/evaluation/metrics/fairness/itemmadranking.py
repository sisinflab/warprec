# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
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
        item_clusters (Tensor): Tensor mapping each item to an item cluster.
        item_counts (Tensor): Tensor of counts of item recommended.
        item_gains (Tensor): Tensor of gains of item recommended.
        n_items (int): Total number of items.

    Args:
        k (int): Cutoff for top-k recommendations.
        train_set (csr_matrix): Sparse matrix of training interactions (users x items).
        *args (Any): The argument list.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.DISCOUNTED_RELEVANCE,
        MetricBlock.TOP_K_INDICES,
    }

    item_clusters: Tensor
    item_counts: Tensor
    item_gains: Tensor
    n_items: int

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        *args: Any,
        item_cluster: Tensor = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.n_items = train_set.shape[1]

        self.register_buffer("item_clusters", item_cluster)

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

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target: Tensor = kwargs.get("discounted_relevance", torch.zeros_like(preds))
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )

        # Item counts
        counts = torch.bincount(
            top_k_indices.flatten(), minlength=self.n_items
        )  # [num_items]

        # Item gains
        rel = torch.zeros_like(preds)
        top_k_values = torch.gather(target, 1, top_k_indices)
        rel.scatter_(1, top_k_indices, top_k_values)  # [batch_size x num_items]

        self.item_counts += counts
        self.item_gains += rel.sum(dim=0)

    def compute(self):
        item_avg_gain_when_recommended = torch.zeros_like(self.item_gains)
        recommended_mask = self.item_counts > 0

        item_avg_gain_when_recommended[recommended_mask] = (
            self.item_gains[recommended_mask] / self.item_counts[recommended_mask]
        )

        num_clusters = self.item_clusters.max().item() + 1

        recommended_item_indices = torch.where(recommended_mask)[0]
        recommended_item_cluster_indices = self.item_clusters[recommended_item_indices]
        recommended_item_avg_gains = item_avg_gain_when_recommended[
            recommended_item_indices
        ]

        cluster_sum_of_avg_item_gains = torch.bincount(
            recommended_item_cluster_indices,
            weights=recommended_item_avg_gains,
            minlength=num_clusters,
        )

        cluster_recommended_item_counts = torch.bincount(
            recommended_item_cluster_indices, minlength=num_clusters
        )

        cluster_mean = torch.zeros(
            num_clusters, dtype=torch.float, device=self.item_gains.device
        )
        mask = cluster_recommended_item_counts > 0
        cluster_mean[mask] = (
            cluster_sum_of_avg_item_gains[mask] / cluster_recommended_item_counts[mask]
        )

        valid_clusters_mask = cluster_recommended_item_counts > 0
        valid_cluster_means = cluster_mean[valid_clusters_mask]

        if valid_cluster_means.numel() < 2:
            mad = torch.tensor(0.0, device=self.item_gains.device)
        else:
            i, j = torch.triu_indices(
                valid_cluster_means.size(0), valid_cluster_means.size(0), offset=1
            )
            diff_matrix = torch.abs(
                valid_cluster_means.unsqueeze(0) - valid_cluster_means.unsqueeze(1)
            )
            differences = diff_matrix[i, j]
            mad = differences.mean()

        return {self.name: mad.item()}

    def reset(self):
        """Resets the metric state."""
        self.item_counts.zero_()
        self.item_gains.zero_()
