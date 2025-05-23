from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.utils.registry import metric_registry
from warprec.evaluation.base_metric import TopKMetric


@metric_registry.register("BiasDisparityBR")
class BiasDisparityBR(TopKMetric):
    """The BiasDisparityBR@K (Bias Disparity - Bias Recommendations) metric.

    This metric computes the disparity between the distribution of recommended items and the global
    item distribution per user cluster, averaged over users in the cluster.

    The metric is computed as:

        BR(u, c) = (P_rec(u, c) / P_rec(u)) / P_global(c)

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
        n_user_effective_clusters (int): The total number of unique user clusters.
        n_user_clusters (int): The total number of unique user clusters, including fallback cluster.
        item_clusters (Tensor): Tensor mapping each item to an item cluster.
        n_item_effective_clusters (int): The total number of unique item clusters.
        n_item_clusters (int): The total number of unique item clusters, including fallback cluster.
        PC (Tensor): Global distribution of items across item clusters.
        category_sum (Tensor): Accumulator tensor of shape counting recommended items per user-item cluster pair.
        total_sum (Tensor): Accumulator tensor counting total recommendations per user cluster.

    Args:
        k (int): The cutoff.
        train_set (csr_matrix): Sparse matrix of training interactions (users x items).
        *args (Any): The argument list.
        user_cluster (Tensor): Lookup tensor of user clusters.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    user_clusters: Tensor
    n_user_effective_clusters: int
    n_user_clusters: int
    item_clusters: Tensor
    n_item_effective_clusters: int
    n_item_clusters: int
    PC: Tensor
    category_sum: Tensor
    total_sum: Tensor

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        *args: Any,
        user_cluster: Tensor = None,
        item_cluster: Tensor = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.train_set = train_set

        self.register_buffer("user_clusters", user_cluster)
        self.n_user_effective_clusters = int(user_cluster.max().item())
        self.n_user_clusters = (
            self.n_user_effective_clusters + 1
        )  # Take into account the zero cluster

        self.register_buffer("item_clusters", item_cluster)
        self.n_item_effective_clusters = int(item_cluster.max().item())
        self.n_item_clusters = (
            self.n_item_effective_clusters + 1
        )  # Take into account the zero cluster

        # Global distribution of items across item clusters
        pc = torch.bincount(item_cluster, minlength=self.n_item_clusters).float()
        pc = pc / float(self.train_set.shape[1])
        self.register_buffer("PC", pc)

        # Initialize accumulators for counts per user cluster and item cluster
        self.add_state(
            "category_sum",
            default=torch.zeros(self.n_user_clusters, self.n_item_clusters),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_sum", default=torch.zeros(self.n_user_clusters), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        start = kwargs.get("start", 0)

        batch_size = preds.size(0)
        device = preds.device

        # Get user indices in global space
        user_indices = torch.arange(start, start + batch_size, device=device)

        # For each user, get top-k item indices by descending score
        topk = torch.topk(preds, self.k, dim=1).indices  # [batch_size x cutoff]

        # Map user indices to user clusters
        user_clusters = self.user_clusters[user_indices]  # [batch_size]

        # Flatten for batch processing
        user_clusters_expanded = (
            user_clusters.unsqueeze(1).expand(-1, self.k).reshape(-1)
        )  # [batch_size * cutoff]
        item_clusters = self.item_clusters[topk.reshape(-1)]  # [batch_size * cutoff]

        # Count occurrences of each (user_cluster, item_cluster) pair
        # Create a 2D histogram by combining indices into a single index
        combined = user_clusters_expanded * self.n_item_clusters + item_clusters
        counts = torch.bincount(
            combined, minlength=self.n_user_clusters * self.n_item_clusters
        ).float()

        # Reshape counts to [n_user_clusters, n_item_clusters]
        counts = counts.reshape(self.n_user_clusters, self.n_item_clusters)

        # Update accumulators
        self.category_sum += counts
        self.total_sum += counts.sum(dim=1)

    def compute(self):
        """Computes the final metric value."""
        # Avoid division by zero
        total_sum_safe = self.total_sum.clone()
        total_sum_safe = total_sum_safe.clamp(min=1)

        # Compute per user cluster distribution of recommended items
        rec_dist = self.category_sum / total_sum_safe.unsqueeze(
            1
        )  # [n_user_clusters x n_item_clusters]

        # Compute bias disparity ratio
        bias_disparity = rec_dist / self.PC.unsqueeze(0)  # broadcast over user clusters

        results = {}
        for uc in range(self.n_user_effective_clusters):
            for ic in range(self.n_item_effective_clusters):
                key = f"BiasDisparityBR_UC{uc + 1}_IC{ic + 1}"
                results[key] = bias_disparity[uc + 1, ic + 1].item()
        return results

    def reset(self):
        """Resets the metric state."""
        self.category_sum.zero_()
        self.total_sum.zero_()
