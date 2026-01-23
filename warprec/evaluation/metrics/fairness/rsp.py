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

    The metric formula is defined as:
        RSP = std(P(R@k | g=g_1), ..., P(R@k | g=g_A)) / mean(P(R@k | g=g_1), ..., P(R@k | g=g_A))

    where:
        - P(R@k | g=g_a) is the probability that an item from group g_a is ranked
          in the top-k recommendations, relative to the pool of items from g_a
          not seen during training.
        - g_a represents an item cluster.
        - A is the total number of item clusters.

    The probability for a cluster g_a is calculated as:
        P(R@k | g=g_a) = (Sum over users u of |{items in top-k for u} AND
            {items in group g_a}|) / (Sum over users u of |{items not in training set for u} AND {items in group g_a}|)

    This simplifies to:
        P(R@k | g=g_a) = (Total count of group g_a items in top-k across all users) /
            (Total count of group g_a items not in training set across all users)

    Matrix computation of the numerator within a batch:
    Given recommendations (preds) and ground truth relevance (target) for a batch of users:
        PREDS (Scores)        TARGETS (Binary Relevance)
    +---+---+---+---+       +---+---+---+---+
    | . | . | . | . |       | 1 | 0 | 1 | 0 |
    | . | . | . | . |       | 0 | 0 | 1 | 1 |
    +---+---+---+---+       +---+---+---+---+
    Item Clusters: [0, 1, 0, 1] (Example mapping item index to cluster ID)

    1. Get top-k recommended item indices for each user.
    2. Create a binary mask 'top_k_mask' where top_k_mask[u, i] = 1 if item i is in u's top-k recommendations, 0 otherwise.
    3. For each item cluster 'c' (0 to n_item_clusters-1):
        a. Sum top_k_mask[u, i] for all users u and items i where item_clusters[i] == c.
            This is the total count of recommended items from cluster c in the batch.
    4. Accumulate these counts across batches into 'cluster_recommendations'.

    The denominator (Total count of group g_a items not in training set across all users)
    is pre-calculated during initialization using the provided training set.

    After processing all batches, compute the per-cluster probabilities:
        pr_c = (total recommended items from cluster c) / (total items from cluster c not in training set)
    Compute RSP = std(pr_0, pr_1, ..., pr_{A-1}) / mean(pr_0, pr_1, ..., pr_{A-1}).
    Handle cases where the denominator is zero or no items exist in the eligible pool for a cluster.

    For further details, please refer to this `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_.

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
