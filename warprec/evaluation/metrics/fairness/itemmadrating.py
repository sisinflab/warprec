# pylint: disable=arguments-differ, unused-argument, line-too-long, duplicate-code
from typing import Any, Set

import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
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
        num_items (int): Number of items in the training set.
        item_clusters (Tensor): Tensor mapping each item to an item cluster.
        item_counts (Tensor): Tensor of counts of item recommended and relevant.
        item_gains (Tensor): Tensor of summed ratings/relevance for item recommended and relevant.

    Args:
        k (int): Cutoff for top-k recommendations.
        num_items (int): Number of items in the training set.
        *args (Any): The argument list.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.TOP_K_VALUES,
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
        *args: Any,
        item_cluster: Tensor = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)

        self.num_items = num_items

        self.register_buffer("item_clusters", item_cluster)

        # Initialize accumulators for counts and gains (relevant items only)
        self.add_state(
            "item_counts",
            torch.zeros(num_items),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "item_gains",
            torch.zeros(num_items),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        top_k_values: Tensor = kwargs.get(
            f"top_{self.k}_values", self.top_k_values_indices(preds, self.k)[0]
        )
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )

        # Get the item indices for the current batch (either full or sampled)
        item_indices = kwargs.get("item_indices", None)
        if item_indices is not None:
            batch_item_indices = item_indices
        else:
            batch_item_indices = (
                torch.arange(preds.shape[1], device=preds.device)
                .unsqueeze(0)
                .repeat(preds.shape[0], 1)
            )

        # Compute relevance mask using local top_k_indices
        batch_size = target.size(0)
        row_indices = torch.arange(batch_size, device=target.device)[:, None].expand(
            -1, self.k
        )
        relevance_mask = target[row_indices, top_k_indices].bool()  # [batch_size x k]

        # Filter out the global item indices for relevant items only
        relevant_top_k_global_indices = torch.masked_select(
            torch.gather(batch_item_indices, 1, top_k_indices), relevance_mask
        )  # [batch_size, k]

        # Filter out the gains and counts for relevant items only
        relevant_top_k_gains = torch.masked_select(
            top_k_values, relevance_mask
        )  # [batch_size, k]

        # Accumulate counts and gains
        self.item_counts.index_add_(
            0, relevant_top_k_global_indices, torch.ones_like(relevant_top_k_gains)
        )
        self.item_gains.index_add_(
            0, relevant_top_k_global_indices, relevant_top_k_gains
        )

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
            mad = torch.tensor(0.0, device=self.item_gains.device).item()
        else:
            i, j = torch.triu_indices(
                valid_cluster_means.size(0), valid_cluster_means.size(0), offset=1
            )
            diff_matrix = torch.abs(
                valid_cluster_means.unsqueeze(0) - valid_cluster_means.unsqueeze(1)
            )
            differences = diff_matrix[i, j]
            mad = differences.mean().item()

        return {self.name: mad}
