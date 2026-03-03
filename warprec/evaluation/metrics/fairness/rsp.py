from typing import Any, Set

import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("RSP")
class RSP(TopKMetric):
    """Ranking-based Statistical Parity (RSP) metric.

    This metric evaluates the fairness of a recommender system by comparing the
    proportion of recommended items from different item clusters (or groups)
    out of the pool of items not seen during training. It calculates the standard
    deviation of these proportions divided by their mean, providing a measure
    of how equally the system recommends items across different groups, regardless
    of relevance in the test set.

    Attributes:
        item_clusters (Tensor): A tensor mapping item index to its cluster ID.
        cluster_recommendations (Tensor): Accumulator for the total count of recommended items per cluster in the top-k.
        denominator_counts (Tensor): Pre-calculated total count of items per cluster not in the training set across all users.
        n_effective_clusters (int): The total number of unique item clusters.
        n_item_clusters (int): The total number of unique item clusters, including fallback cluster.
        user_interactions (Tensor): Accumulator for counting how many times each user has been evaluated.

    Args:
        k (int): Cutoff for top-k recommendations.
        num_users (int): Number of users in the training set.
        item_interactions (Tensor): Tensor containing counts of item interactions in the training set.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.TOP_K_INDICES,
        MetricBlock.VALID_USERS,
    }

    item_clusters: Tensor
    cluster_recommendations: Tensor
    denominator_counts: Tensor
    n_effective_clusters: int
    n_item_clusters: int
    user_interactions: Tensor

    def __init__(
        self,
        k: int,
        num_users: int,
        item_interactions: Tensor,
        item_cluster: Tensor = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.register_buffer("item_clusters", item_cluster)
        self.n_effective_clusters = int(item_cluster.max().item())
        self.n_item_clusters = self.n_effective_clusters + 1

        # Count cluster of items in the catalog
        self.register_buffer(
            "cluster_item_counts",
            torch.bincount(item_cluster, minlength=self.n_item_clusters).float(),
        )

        # Global count of items per cluster in the training set
        cluster_train_counts = torch.zeros(
            self.n_item_clusters, dtype=torch.float, device=item_cluster.device
        )
        cluster_train_counts.index_add_(0, item_cluster, item_interactions.float())
        self.register_buffer("cluster_train_interaction_counts", cluster_train_counts)

        # Accumulators
        self.add_state(
            "cluster_recommendations",
            torch.zeros(self.n_item_clusters, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "user_interactions",
            default=torch.zeros(num_users, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        users = kwargs.get("valid_users")
        top_k_indices = kwargs.get(f"top_{self.k}_indices")

        # Remap top_k_indices to global
        item_indices = kwargs.get("item_indices")
        top_k_indices = self.remap_indices(top_k_indices, item_indices)

        # Accumulate cluster recommendations for numerator
        flat_indices = top_k_indices.flatten()
        rec_clusters = self.item_clusters[flat_indices]
        batch_rec_counts = torch.bincount(
            rec_clusters, minlength=self.n_item_clusters
        ).float()
        self.cluster_recommendations += batch_rec_counts

        # Accumulate user interactions for denominator
        self.user_interactions.index_add_(0, user_indices, users.float())

    def compute(self):
        # Compute total interactions across all users
        total_interactions = self.user_interactions.sum()

        if total_interactions == 0:
            return {self.name: 0.0}

        # Total potential items per cluster not in training set
        total_potential = total_interactions * self.cluster_item_counts

        # Estimate masked items per cluster
        num_total_users = self.user_interactions.size(0)
        scaling_factor = total_interactions / num_total_users
        estimated_masked_items = scaling_factor * self.cluster_train_interaction_counts

        # Final denominator counts
        denominator_counts = total_potential - estimated_masked_items

        # Safety clamp to avoid negative values
        denominator_counts = torch.clamp(denominator_counts, min=0.0)

        # Valid clusters for computation
        valid_mask = denominator_counts > 0

        if not valid_mask.any():
            return {self.name: 0.0}

        # Compute probabilities per cluster
        probs = torch.zeros_like(self.cluster_recommendations)
        probs[valid_mask] = (
            self.cluster_recommendations[valid_mask] / denominator_counts[valid_mask]
        )

        valid_probs = probs[valid_mask]

        if valid_probs.numel() <= 1:
            std_prob = 0.0
            mean_prob = 1.0
        else:
            std_prob = torch.std(valid_probs, unbiased=False).item()
            mean_prob = torch.mean(valid_probs).item()

        results = {}

        # Populate per-cluster probability
        for ic in range(1, self.n_effective_clusters + 1):
            key = f"{self.name}_IC{ic}"
            if valid_mask[ic]:
                results[key] = probs[ic].item()
            else:
                results[key] = float("nan")

        # Aggregate Score
        if mean_prob == 0:
            results[self.name] = 0.0
        else:
            results[self.name] = std_prob / mean_prob

        return results
