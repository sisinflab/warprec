from typing import Any, Set

import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("UserMADRating")
class UserMADRating(UserAverageTopKMetric):
    """User MAD Rating (UserMADRating) metric.

    This metric measures the disparity in the average rating/score received by users
    across different user clusters, considering the average rating of their top-k
    recommended items. It computes the Mean Absolute Deviation (MAD) of the average
    per-user average top-k rating scores per user cluster. The MAD is computed as the mean
    of absolute differences between every pair of cluster-level averages.

    Attributes:
        user_clusters (Tensor): Tensor mapping each user to an user cluster.

    Args:
        k (int): Cutoff for top-k recommendations.
        num_users (int): Number of users in the training set.
        user_cluster (Tensor): Lookup tensor of user clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.TOP_K_VALUES,
        MetricBlock.VALID_USERS,
    }

    user_clusters: Tensor

    def __init__(
        self,
        k: int,
        num_users: int,
        user_cluster: Tensor,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k=k, num_users=num_users, dist_sync_on_step=dist_sync_on_step)

        self.register_buffer("user_clusters", user_cluster)
        self.n_user_clusters = int(user_cluster.max().item()) + 1

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        top_k_values = kwargs.get(f"top_{self.k}_values")

        # Average score of the top-k items for each user
        return top_k_values.mean(dim=1)

    def compute(self):
        # Calculate average rating per user
        mask = self.user_interactions > 0

        if not mask.any():
            return {self.name: torch.tensor(0.0)}

        user_vals = torch.zeros_like(self.scores)
        user_vals[mask] = self.scores[mask] / self.user_interactions[mask]

        # Aggregate per cluster
        sum_cluster = torch.zeros(self.n_user_clusters, device=self.device)
        count_cluster = torch.zeros(
            self.n_user_clusters, dtype=torch.long, device=self.device
        )

        sum_cluster.scatter_add_(0, self.user_clusters, user_vals)
        count_cluster.scatter_add_(0, self.user_clusters, mask.long())

        # Mean per cluster
        valid_clusters = count_cluster > 0

        if not valid_clusters.any():
            return {self.name: torch.tensor(0.0)}

        mean_cluster = sum_cluster[valid_clusters] / count_cluster[valid_clusters]

        # Pairwise absolute differences (MAD)
        m = mean_cluster.numel()
        if m < 2:
            mad = torch.tensor(0.0, device=self.device)
        else:
            diffs = (mean_cluster.unsqueeze(0) - mean_cluster.unsqueeze(1)).abs()
            pairwise_sum = diffs.triu(diagonal=1).sum()
            num_pairs = m * (m - 1) / 2
            mad = pairwise_sum / num_pairs

        return {self.name: mad.item()}
