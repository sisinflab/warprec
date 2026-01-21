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
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
        MetricBlock.TOP_K_INDICES,
        MetricBlock.TOP_K_VALUES,
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

        self.register_buffer("item_clusters", item_cluster)
        self.n_item_clusters = int(item_cluster.max().item()) + 1

        # Initialize accumulators
        self.add_state("item_counts", torch.zeros(num_items), dist_reduce_fx="sum")
        self.add_state("item_gains", torch.zeros(num_items), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        top_k_values = kwargs.get(f"top_{self.k}_values")
        top_k_indices = kwargs.get(f"top_{self.k}_indices")
        top_k_rel = kwargs.get(f"top_{self.k}_binary_relevance")

        # Create relevance mask (only consider relevant items)
        rel_mask = top_k_rel.bool()

        # Handle sampled item indices if provided (map local batch indices to global item IDs)
        item_indices = kwargs.get("item_indices")
        if item_indices is not None:
            top_k_indices = torch.gather(item_indices, 1, top_k_indices)

        # Filter only Relevant items (True Positives)
        # We select elements from global_indices and top_k_values where relevance_mask is True
        relevant_indices = torch.masked_select(top_k_indices, rel_mask)
        relevant_scores = torch.masked_select(top_k_values, rel_mask)

        if relevant_indices.numel() > 0:
            # Accumulate counts (1 for every relevant appearance)
            self.item_counts.index_add_(
                0, relevant_indices, torch.ones_like(relevant_scores)
            )

            # Accumulate gains (predicted scores of relevant items)
            self.item_gains.index_add_(0, relevant_indices, relevant_scores)

    def compute(self):
        # Compute average rating per item (only for recommended and relevant items)
        recommended_mask = self.item_counts > 0

        if not recommended_mask.any():
            return {self.name: torch.tensor(0.0)}

        item_avg_ratings = torch.zeros_like(self.item_gains)
        item_avg_ratings[recommended_mask] = (
            self.item_gains[recommended_mask] / self.item_counts[recommended_mask]
        )

        # Aggregate per cluster
        rec_indices = torch.where(recommended_mask)[0]
        rec_clusters = self.item_clusters[rec_indices]
        rec_ratings = item_avg_ratings[rec_indices]

        cluster_sum_ratings = torch.zeros(self.n_item_clusters, device=self.device)
        cluster_counts = torch.zeros(self.n_item_clusters, device=self.device)

        cluster_sum_ratings.index_add_(0, rec_clusters, rec_ratings)
        cluster_counts.index_add_(0, rec_clusters, torch.ones_like(rec_ratings))

        # Compute Mean per cluster
        valid_clusters = cluster_counts > 0

        if not valid_clusters.any():
            return {self.name: torch.tensor(0.0)}

        cluster_means = (
            cluster_sum_ratings[valid_clusters] / cluster_counts[valid_clusters]
        )

        # Compute MAD
        if cluster_means.numel() < 2:
            mad = torch.tensor(0.0, device=self.device)
        else:
            diffs = (cluster_means.unsqueeze(0) - cluster_means.unsqueeze(1)).abs()
            pairwise_sum = diffs.triu(diagonal=1).sum()
            num_pairs = cluster_means.numel() * (cluster_means.numel() - 1) / 2
            mad = pairwise_sum / num_pairs

        return {self.name: mad.item()}
