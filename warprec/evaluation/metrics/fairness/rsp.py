# pylint: disable=arguments-differ, unused-argument, line-too-long
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
        P(R@k | g=g_a) = (Sum over users u of |{items in top-k for u} AND {items in group g_a}|) / (Sum over users u of |{items not in training set for u} AND {items in group g_a}|)

    This simplifies to:
        P(R@k | g=g_a) = (Total count of group g_a items in top-k across all users) / (Total count of group g_a items not in training set across all users)

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
        a. Sum top_k_mask[u, i] for all users u and items i where item_clusters[i] == c. This is the total count of recommended items from cluster c in the batch.
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

    Args:
        k (int): Cutoff for top-k recommendations.
        num_users (int): Number of users in the training set.
        item_indices (Tensor): Indices of items in the training set used to calculate the denominator counts.
        *args (Any): The argument list.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {MetricBlock.TOP_K_INDICES}

    item_clusters: Tensor
    cluster_recommendations: Tensor
    denominator_counts: Tensor
    n_effective_clusters: int
    n_item_clusters: int

    def __init__(
        self,
        k: int,
        num_users: int,
        item_indices: Tensor,
        *args: Any,
        item_cluster: Tensor = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.register_buffer("item_clusters", item_cluster)
        self.n_effective_clusters = int(item_cluster.max().item())
        self.n_item_clusters = (
            self.n_effective_clusters + 1
        )  # Take into account the zero cluster

        # Calculate |g_a| for each cluster g_a
        cluster_item_counts = torch.bincount(
            self.item_clusters, minlength=self.n_item_clusters
        ).float()  # [num_clusters]

        # Calculate total count of training interactions whose item is in group g_a
        train_item_clusters = self.item_clusters[item_indices]

        # Count occurrences of each cluster ID among training items
        cluster_train_interaction_counts = torch.zeros(
            self.n_item_clusters, dtype=torch.float
        )
        cluster_train_interaction_counts.scatter_add_(
            0,
            train_item_clusters,
            torch.ones_like(train_item_clusters, dtype=torch.float),
        )  # Sum counts per cluster

        # Calculate the denominator for each cluster
        total_potential_items_per_cluster = num_users * cluster_item_counts
        denominator_counts = (
            total_potential_items_per_cluster - cluster_train_interaction_counts
        )  # [n_item_clusters]

        self.register_buffer("denominator_counts", denominator_counts)
        self.add_state(
            "cluster_recommendations",
            torch.zeros(self.n_item_clusters, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )
        batch_size = preds.shape[0]

        # Create a mask for items in the top-k recommendations
        top_k_mask = torch.zeros_like(preds, dtype=torch.bool, device=preds.device)
        top_k_mask.scatter_(
            dim=1, index=top_k_indices, value=True
        )  # [batch_size x num_items]

        # Accumulate counts per cluster for recommended items (regardless of relevance)
        # Flatten the batch mask and repeat cluster IDs for each user
        flat_top_k_mask_float = top_k_mask.float().flatten()  # [batch_size * num_items]
        flat_item_clusters = self.item_clusters.repeat(
            batch_size
        )  # [batch_size * num_items]

        # Initialize batch accumulator for recommendations
        batch_rec_counts = torch.zeros(
            self.n_item_clusters, dtype=torch.float, device=preds.device
        )

        # Use scatter_add to sum values based on cluster index
        batch_rec_counts.scatter_add_(0, flat_item_clusters, flat_top_k_mask_float)

        # Update the state variable
        self.cluster_recommendations += batch_rec_counts

    def compute(self):
        """Computes the final value of the metric."""
        # Find clusters with a non-zero denominator (i.e., eligible items in the test pool)
        valid_cluster_mask = self.denominator_counts > 0

        # If no clusters have eligible items in the test pool, RSP is undefined, return 0.0
        if not valid_cluster_mask.any():
            return {self.name: torch.tensor(0.0).item()}

        # Compute probabilities only for clusters with eligible items
        # pr_c = (recommended in cluster c) / (eligible items in cluster c)
        valid_cluster_recommendations = self.cluster_recommendations[valid_cluster_mask]
        valid_denominator_counts = self.denominator_counts[valid_cluster_mask]
        cluster_probabilities = valid_cluster_recommendations / valid_denominator_counts

        # If there's only one valid cluster, std dev is 0, so RSP is 0.0
        if cluster_probabilities.numel() <= 1:
            return {self.name: torch.tensor(0.0).item()}

        # Calculate RSP = std(probs) / mean(probs)
        mean_prob = torch.mean(cluster_probabilities)

        # Handle case where mean probability is zero
        # (e.g., valid clusters had eligible items, but none were recommended in top-k)
        if mean_prob == 0:
            return {self.name: torch.tensor(0.0).item()}

        std_prob = torch.std(
            cluster_probabilities, unbiased=False
        )  # Use population standard deviation as in original formula

        rsp_score = std_prob / cluster_probabilities

        results = {}
        for ic in range(self.n_effective_clusters):
            key = f"{self.name}_IC{ic + 1}"
            results[key] = rsp_score[ic].item()

        results[self.name] = (std_prob / mean_prob).item()
        return results
