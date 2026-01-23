from typing import Any, Set

import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
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
        P(R@k | g=g_a, y=1) = (Sum over users u of |{relevant items in top-k for u} AND
            {items in group g_a}|) / (Sum over users u of |{relevant items for u} AND {items in group g_a}|)

    This simplifies to:
        P(R@k | g=g_a, y=1) = (Total count of relevant group g_a items in top-k across all users) /
            (Total count of relevant group g_a items across all users)

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
        a. Sum rel[u, i] for all users u and items i where item_clusters[i] == c.
            This is the total count of relevant recommended items from cluster c in the batch.
        b. Sum target[u, i] for all users u and items i where item_clusters[i] == c.
            This is the total count of relevant items from cluster c in the ground truth for the batch.
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
        *args (Any): The argument list.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.TOP_K_INDICES,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    item_clusters: Tensor
    cluster_recommendations: Tensor
    cluster_total_items: Tensor
    n_effective_clusters: int
    n_item_clusters: int

    def __init__(
        self,
        k: int,
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
            torch.zeros(self.n_item_clusters),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "cluster_total_items",
            torch.zeros(self.n_item_clusters),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, **kwargs: Any):
        target = kwargs.get("binary_relevance")
        top_k_indices = kwargs.get(f"top_{self.k}_indices")
        top_k_rel = kwargs.get(f"top_{self.k}_binary_relevance")
        item_indices = kwargs.get("item_indices")

        # Remap top_k_indices to global
        item_indices = kwargs.get("item_indices")
        top_k_indices = self.remap_indices(top_k_indices, item_indices)

        # Identify Global Indices for Recommendations
        if item_indices is not None:
            rows, cols = target.nonzero(as_tuple=True)
            positive_indices_global = item_indices[rows, cols]
        else:
            _, positive_indices_global = target.nonzero(as_tuple=True)

        # Identify Relevant Recommended Items
        rel_mask = top_k_rel > 0
        relevant_rec_indices_global = top_k_indices[rel_mask]

        # Map to Clusters
        rec_clusters = self.item_clusters[relevant_rec_indices_global]
        gt_clusters = self.item_clusters[positive_indices_global]

        # Accumulate Counts
        batch_rec_counts = torch.bincount(
            rec_clusters, minlength=self.n_item_clusters
        ).float()

        batch_total_counts = torch.bincount(
            gt_clusters, minlength=self.n_item_clusters
        ).float()

        self.cluster_recommendations += batch_rec_counts
        self.cluster_total_items += batch_total_counts

    def compute(self):
        # Mask for clusters that exist in the ground truth
        # We assume cluster 0 is padding/unknown and usually ignore it if it has no items
        valid_mask = self.cluster_total_items > 0

        if not valid_mask.any():
            return {self.name: torch.tensor(0.0).item()}

        # Calculate probabilities for ALL clusters (keep 0 for invalid ones to maintain index alignment)
        # Avoid division by zero
        safe_denom = self.cluster_total_items.clone()
        safe_denom[~valid_mask] = 1.0

        probs = self.cluster_recommendations / safe_denom
        probs[~valid_mask] = 0.0  # Ensure invalid clusters are 0

        # Calculate global stats based ONLY on valid clusters
        valid_probs = probs[valid_mask]

        if valid_probs.numel() <= 1:
            std_prob = torch.tensor(0.0)
            mean_prob = torch.tensor(1.0)  # Avoid div/0
        else:
            std_prob = torch.std(valid_probs, unbiased=False)
            mean_prob = torch.mean(valid_probs)

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
            results[self.name] = (std_prob / mean_prob).item()

        return results
