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
        cluster_available (Tensor): Accumulator for the items of each cluster
            that were on offer to be recommended.
        cluster_recommendations (Tensor): Accumulator for the total count of recommended items per cluster in the top-k.
        denominator_counts (Tensor): Pre-calculated total count of items per cluster
            not in the training set across all users.
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
    cluster_available: Tensor
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

        # Accumulators
        self.add_state(
            "cluster_available",
            torch.zeros(self.n_item_clusters, dtype=torch.float),
            dist_reduce_fx="sum",
        )
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

        # A cluster's rate is how often it was recommended out of how often it
        # could have been. The evaluator has already put everything unavailable
        # beyond reach — items the user saw in training, and anything outside a
        # restricted candidate set — so what stays finite is exactly what was on
        # offer. Estimating it instead, by spreading the training mass evenly
        # across users, gave rates above one whenever the users being scored had
        # shorter histories than the population average.
        offered = torch.isfinite(preds)
        if item_indices is None:
            columns = torch.arange(preds.size(1), device=preds.device).expand_as(preds)
        else:
            columns = item_indices

        self.cluster_available += torch.bincount(
            self.item_clusters[columns[offered]], minlength=self.n_item_clusters
        ).float()

        # Kept so a run can still report how many users were scored.
        self.user_interactions.index_add_(0, user_indices, users.float())

    def compute(self):
        # Compute total interactions across all users
        total_interactions = self.user_interactions.sum()

        if total_interactions == 0:
            return {self.name: 0.0}

        # What was actually on offer, counted rather than estimated.
        denominator_counts = self.cluster_available

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
