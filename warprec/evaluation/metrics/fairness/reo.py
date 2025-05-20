# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("REO")
class REO(TopKMetric):
    """Ranking-based Equal Opportunity (REO) metric.

    This metric evaluates the fairness of a recommender system by comparing the
    proportion of recommended items from different item clusters (or groups)
    among the relevant items in the ground truth. It calculates the standard
    deviation of these proportions divided by their mean, providing a measure
    of how equally the system recommends relevant items across different groups.

    The metric formula is defined as:
        REO = std(P(R@k | g=g_1, y=1), ..., P(R@k | g=g_A, y=1)) / mean(P(R@k | g=g_1, y=1), ..., P(R@k | g=g_A, y=1))

    where:
        - P(R@k | g=g_a, y=1) is the probability that a relevant item from group g_a
          is ranked in the top-k recommendations.
        - g_a represents an item cluster.
        - A is the total number of item clusters.

    The probability for a cluster g_a is calculated as:
        P(R@k | g=g_a, y=1) = (Sum over users u of |{relevant items in top-k for u} AND {items in group g_a}|) / (Sum over users u of |{relevant items for u} AND {items in group g_a}|)

    This simplifies to:
        P(R@k | g=g_a, y=1) = (Total count of relevant group g_a items in top-k across all users) / (Total count of relevant group g_a items across all users)

    Matrix computation of the metric within a batch:
    Given recommendations (preds) and ground truth relevance (target) for a batch of users:
        PREDS (Scores)        TARGETS (Binary Relevance)
    +---+---+---+---+       +---+---+---+---+
    | . | . | . | . |       | 1 | 0 | 1 | 0 |
    | . | . | . | . |       | 0 | 0 | 1 | 1 |
    +---+---+---+---+       +---+---+---+---+
    Item Clusters: [0, 1, 0, 1] (Example mapping item index to cluster ID)

    1. Get top-k recommended item indices for each user.
    2. Create a binary matrix 'rel' where rel[u, i] = 1 if item i is relevant to user u AND i is in u's top-k recommendations, 0 otherwise.
    3. For each item cluster 'c' (0 to n_item_clusters-1):
        a. Sum rel[u, i] for all users u and items i where item_clusters[i] == c. This is the total count of relevant recommended items from cluster c in the batch.
        b. Sum target[u, i] for all users u and items i where item_clusters[i] == c. This is the total count of relevant items from cluster c in the ground truth for the batch.
    4. Accumulate these counts across batches.

    After processing all batches, compute the per-cluster probabilities:
        pr_c = (total relevant recommended items from cluster c) / (total relevant items from cluster c)
    Compute REO = std(pr_0, pr_1, ..., pr_{A-1}) / mean(pr_0, pr_1, ..., pr_{A-1}).
    Handle cases where the denominator is zero or no relevant items exist for a cluster.

    For further details, please refer to this `paper <https://dl.acm.org/doi/abs/10.1145/3397271.3401177>`_.

    Attributes:
        item_clusters (Tensor): A tensor mapping item index to its cluster ID.
        cluster_recommendations (Tensor): Accumulator for the total count of relevant recommended items per cluster.
        cluster_total_items (Tensor): Accumulator for the total count of relevant items per cluster in the ground truth.
        n_effective_clusters (int): The total number of unique item clusters.
        n_item_clusters (int): The total number of unique item clusters, including fallback cluster.

    Args:
        k (int): Cutoff for top-k recommendations.
        train_set (csr_matrix): Sparse matrix of training interactions (users x items). (Used for initialization, not directly in update/compute logic in this implementation).
        *args (Any): The argument list.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    item_clusters: Tensor
    cluster_recommendations: Tensor
    cluster_total_items: Tensor
    n_effective_clusters: int
    n_item_clusters: int

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
        self.register_buffer("item_clusters", item_cluster)
        self.n_effective_clusters = int(item_cluster.max().item())
        self.n_item_clusters = (
            self.n_effective_clusters + 1
        )  # Take into account the zero cluster

        # Per-cluster accumulators
        self.add_state(
            "cluster_recommendations",
            torch.zeros(self.n_item_clusters, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "cluster_total_items",
            torch.zeros(self.n_item_clusters, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target = self.binary_relevance(target)
        batch_size = preds.shape[0]
        top_k_indices = torch.topk(preds, self.k, dim=1).indices  # [batch_size x k]

        # Create a mask for items in the top-k recommendations
        top_k_mask = torch.zeros_like(preds, dtype=torch.bool)
        top_k_mask.scatter_(
            dim=1, index=top_k_indices, value=True
        )  # [batch_size x num_items]

        # Retrieve relevant items
        rel = target * top_k_mask.int()  # [batch_size x num_items]

        # Accumulate counts per cluster for relevant recommended items
        # Flatten the batch tensors and repeat cluster IDs for each user
        flat_relevant_recommended = rel.flatten()  # [batch_size * num_items]
        flat_target = target.flatten()  # [batch_size * num_items]
        flat_item_clusters = self.item_clusters.repeat(
            batch_size
        )  # [batch_size * num_items]

        # Initialize batch accumulators
        batch_rec_counts = torch.zeros(
            self.n_item_clusters, dtype=torch.float, device=preds.device
        )
        batch_total_counts = torch.zeros(
            self.n_item_clusters, dtype=torch.float, device=preds.device
        )

        # Use scatter_add to sum values based on cluster index
        batch_rec_counts.scatter_add_(0, flat_item_clusters, flat_relevant_recommended)
        batch_total_counts.scatter_add_(0, flat_item_clusters, flat_target)

        # Update the state variables
        self.cluster_recommendations += batch_rec_counts
        self.cluster_total_items += batch_total_counts

    def compute(self):
        """Computes the final value of the metric."""
        # Find clusters with relevant items in the ground truth
        valid_cluster_mask = self.cluster_total_items > 0

        # If no relevant items exist in any cluster, REO is undefined, return 0.0
        if not valid_cluster_mask.any():
            return {self.name: torch.tensor(0.0).item()}

        # Compute probabilities only for clusters with relevant items
        # pr_c = (relevant recommended in cluster c) / (total relevant in cluster c)
        valid_cluster_recommendations = self.cluster_recommendations[valid_cluster_mask]
        valid_cluster_total_items = self.cluster_total_items[valid_cluster_mask]
        cluster_probabilities = (
            valid_cluster_recommendations / valid_cluster_total_items
        )

        # If there's only one valid cluster, std dev is 0, so REO is 0.0
        if cluster_probabilities.numel() <= 1:
            return {self.name: torch.tensor(0.0).item()}

        # Calculate mean prob for later
        mean_prob = torch.mean(cluster_probabilities)

        # Handle case where mean probability is zero
        # (e.g., valid clusters had relevant items, but none were recommended in top-k)
        if mean_prob == 0:
            return {self.name: torch.tensor(0.0).item()}

        std_prob = torch.std(
            cluster_probabilities, unbiased=False
        )  # Use population standard deviation as in original formula

        reo_score = std_prob / cluster_probabilities

        results = {}
        for ic in range(self.n_effective_clusters):
            key = f"{self.name}_IC{ic + 1}"
            results[key] = reo_score[ic].item()

        results[self.name] = (std_prob / mean_prob).item()
        return results

    def reset(self):
        """Resets the metric state."""
        self.cluster_recommendations.zero_()
        self.cluster_total_items.zero_()
