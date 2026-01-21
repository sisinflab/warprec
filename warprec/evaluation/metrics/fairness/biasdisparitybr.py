from typing import Any, Set

import torch
from torch import Tensor

from warprec.utils.registry import metric_registry
from warprec.utils.enums import MetricBlock
from warprec.evaluation.metrics.base_metric import TopKMetric


@metric_registry.register("BiasDisparityBR")
class BiasDisparityBR(TopKMetric):
    """The BiasDisparityBR@K (Bias Disparity - Bias Recommendations) metric.

    This metric computes the disparity between the distribution of recommended items and the global
    item distribution per user cluster, averaged over users in the cluster.

    The metric is computed as:

        BiasDisparityBR(u, c) = (P_rec(u, c) / P_rec(u)) / P_global(c)

    where:
        - u is a user cluster index,
        - c is an item cluster index,
        - P_rec(u, c) is the proportion of recommended items from cluster c to users in cluster u,
        - P_rec(u) is the total number of recommendations to users in cluster u,
        - P_global(c) is the global proportion of items in cluster c.

    A value > 1 indicates over-recommendation of items from cluster c to user cluster u,
    while a value < 1 indicates under-recommendation.

    For further details, please refer to this `link <https://arxiv.org/pdf/1811.01461>`_.

    Attributes:
        user_clusters (Tensor): Tensor mapping each user to a user cluster.
        item_clusters (Tensor): Tensor mapping each item to an item cluster.
        PC (Tensor): Global distribution of items across item clusters.
        category_sum (Tensor): Accumulator tensor of shape counting recommended items per user-item cluster pair.
        total_sum (Tensor): Accumulator tensor counting total recommendations per user cluster.

    Args:
        k (int): The cutoff.
        num_items (int): Number of items in the training set.
        user_cluster (Tensor): Lookup tensor of user clusters.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {MetricBlock.TOP_K_INDICES}

    user_clusters: Tensor
    item_clusters: Tensor
    PC: Tensor
    category_sum: Tensor
    total_sum: Tensor

    def __init__(
        self,
        k: int,
        num_items: int,
        user_cluster: Tensor,
        item_cluster: Tensor,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k=k, dist_sync_on_step=dist_sync_on_step)

        # Register static buffers
        self.register_buffer("user_clusters", user_cluster)
        self.register_buffer("item_clusters", item_cluster)

        self.n_user_effective_clusters = int(user_cluster.max().item())
        self.n_user_clusters = self.n_user_effective_clusters + 1

        self.n_item_effective_clusters = int(item_cluster.max().item())
        self.n_item_clusters = self.n_item_effective_clusters + 1

        # Global distribution of items (P_global)
        pc = torch.bincount(item_cluster, minlength=self.n_item_clusters).float()
        pc = pc / float(num_items)
        self.register_buffer("PC", pc)

        # Accumulators
        self.add_state(
            "category_sum",
            default=torch.zeros(self.n_user_clusters, self.n_item_clusters),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_sum", default=torch.zeros(self.n_user_clusters), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        top_k_indices = kwargs.get(f"top_{self.k}_indices")

        # Handle sampled item indices if provided (map local batch indices to global item IDs)
        item_indices = kwargs.get("item_indices")
        if item_indices is not None:
            top_k_indices = torch.gather(item_indices, 1, top_k_indices)

        # Get User Clusters (expanded to match top-k shape)
        # user_indices: [Batch] -> [Batch, 1] -> [Batch, K] -> [Batch * K]
        batch_user_clusters = self.user_clusters[user_indices]
        flat_user_clusters = (
            batch_user_clusters.unsqueeze(1).expand(-1, self.k).reshape(-1)
        )

        # Get Item Clusters
        flat_item_clusters = self.item_clusters[top_k_indices.reshape(-1)]

        # Vectorized Counting using bincount on flattened 2D coordinates
        # Index = user_cluster * n_item_cols + item_cluster
        combined_indices = (
            flat_user_clusters * self.n_item_clusters + flat_item_clusters
        )

        counts = torch.bincount(
            combined_indices, minlength=self.n_user_clusters * self.n_item_clusters
        ).float()

        # Reshape back to matrix form
        counts_matrix = counts.reshape(self.n_user_clusters, self.n_item_clusters)

        # Update states
        self.category_sum += counts_matrix
        self.total_sum += counts_matrix.sum(dim=1)

    def compute(self):
        # P_rec(u, c) / P_rec(u) / P_global(c)
        safe_total = self.total_sum.unsqueeze(1).clamp(min=1.0)

        bias_rec = (self.category_sum / safe_total) / self.PC.unsqueeze(0)

        results = {}
        for uc in range(self.n_user_effective_clusters):
            for ic in range(self.n_item_effective_clusters):
                # +1 because cluster 0 is usually padding/unknown
                key = f"{self.name}_UC{uc + 1}_IC{ic + 1}"
                results[key] = bias_rec[uc + 1, ic + 1].item()
        return results
