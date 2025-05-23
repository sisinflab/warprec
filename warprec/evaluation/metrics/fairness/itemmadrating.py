# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("ItemMADRating")
class ItemMADRating(TopKMetric):
    """Item MAD Rating (ItemMADRating) metric.

    This metric measures the disparity in the average rating received by items
    across different item clusters, considering only the items that were recommended
    and were relevant to the user. It computes the Mean Absolute Deviation (MAD)
    of the average rating per item cluster. The goal is to evaluate whether some item
    clusters receive consistently higher or lower average ratings when they are
    successfully recommended (i.e., recommended to a relevant user).

    Formally, the metric is defined as:

        MAD = mean_{c_1, c_2} (|mean_avg_rating(c_1) - mean_avg_rating(c_2)|)

    where:
        - mean_avg_rating(c) is the average of the per-item average ratings for all
          items in cluster c that were recommended and relevant at least once.
        - The per-item average rating is the sum of ratings an item received when
          recommended to a relevant user, divided by the count of times it was
          recommended to a relevant user.

    This metric is useful to detect disparities in the quality (in terms of received
    rating/relevance) of recommended items across clusters (e.g., genres, popularity buckets),
    specifically focusing on items that were relevant and successfully recommended.

    The metric tracks the sum of ratings and the count of recommendations for relevant items only
    for each item. These per-item statistics are then aggregated by cluster to compute
    cluster-level means of these average item ratings, and their deviation is computed.

    For further details on the concept, please refer to this `link <https://dl.acm.org/doi/abs/10.1145/3269206.3271795>`_.

    Attributes:
        item_clusters (Tensor): Tensor mapping each item to an item cluster.
        item_counts (Tensor): Tensor of counts of item recommended and relevant.
        item_gains (Tensor): Tensor of summed ratings/relevance for item recommended and relevant.
        n_items (int): Total number of items.

    Args:
        k (int): Cutoff for top-k recommendations.
        train_set (csr_matrix): Sparse matrix of training interactions (users x items).
        *args (Any): The argument list.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

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

        # Initialize accumulators for counts and gains (relevant items only)
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
        target = kwargs.get("binary_relevance", torch.zeros_like(preds))

        # Item counts
        top_k_scores, top_k_indices = torch.topk(preds, self.k, dim=1)
        counts = torch.bincount(
            top_k_indices.flatten(), minlength=self.n_items
        )  # [num_items]

        # Item gains
        batch_size = target.size(0)
        row_indices = torch.arange(batch_size, device=target.device)[:, None].expand(
            -1, self.k
        )
        relevance_mask = target[row_indices, top_k_indices]  # [batch_size x k]

        # Multiply scores by relevance mask (0 for non-relevant items)
        gains = top_k_scores * relevance_mask  # [batch_size x k]

        # Scatter gains to full item dimension and sum
        gain_matrix = torch.zeros_like(preds)
        gain_matrix.scatter_(1, top_k_indices, gains)

        self.item_gains += gain_matrix.sum(dim=0)
        self.item_counts += counts

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
