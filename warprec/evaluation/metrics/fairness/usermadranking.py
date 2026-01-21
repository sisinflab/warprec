from typing import Any, Set, Tuple

import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("UserMADRanking")
class UserMADRanking(UserAverageTopKMetric):
    """User MAD Ranking (UserMADRanking) metric.

    This metric measures the disparity in user exposure across different user clusters
    in the top-k recommendations, by computing the Mean Absolute Deviation (MAD)
    of the average per-user nDCG scores per cluster. The MAD is computed as the mean
    of absolute differences between every pair of cluster-level averages.

    Formally, the metric is defined as:

        MAD = mean_c(|mean_gain(c) - mean_global|)

    where:
        - mean_gain(c) is the average discounted gain of user in cluster c,
        - mean_global is the average of mean_gain(c) over all user clusters.

    This metric is useful to detect disparities in ranking quality across clusters (e.g., genres, popularity buckets).

    The metric uses a discounted relevance model (e.g., log-based) applied to the top-k predictions,

    For further details, please refer to this `link <https://link.springer.com/article/10.1007/s11257-020-09285-1>`_.

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
        MetricBlock.DISCOUNTED_RELEVANCE,
        MetricBlock.TOP_K_DISCOUNTED_RELEVANCE,
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

    def unpack_inputs(
        self, preds: Tensor, **kwargs: Any
    ) -> Tuple[Tensor, Tensor, Tensor]:
        target = kwargs.get("discounted_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel = kwargs.get(
            f"top_{self.k}_discounted_relevance",
            self.top_k_relevance(preds, target, self.k),
        )
        return target, users, top_k_rel

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        # Gather relevance at top-k (DCG component)
        dcg_score = self.dcg(top_k_rel)

        # Compute ideal relevance (IDCG component)
        ideal_rel = torch.topk(target, self.k, dim=1, largest=True, sorted=True).values
        idcg_score = self.dcg(ideal_rel).clamp(min=1e-10)

        # nDCG per user
        return (dcg_score / idcg_score).nan_to_num(0)

    def compute(self):
        # Calculate average nDCG per user
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

        # Scatter add to sum values for each cluster
        sum_cluster.scatter_add_(0, self.user_clusters, user_vals)
        count_cluster.scatter_add_(0, self.user_clusters, mask.long())

        # Mean per cluster
        # Filter out clusters with no users
        valid_clusters = count_cluster > 0

        if not valid_clusters.any():
            return {self.name: torch.tensor(0.0)}

        mean_cluster = sum_cluster[valid_clusters] / count_cluster[valid_clusters]

        # Pairwise absolute differences (MAD)
        m = mean_cluster.numel()
        if m < 2:
            mad = torch.tensor(0.0, device=self.device)
        else:
            # Vectorized pairwise diffs
            # [m, 1] - [1, m] -> [m, m] matrix of differences
            diffs = (mean_cluster.unsqueeze(0) - mean_cluster.unsqueeze(1)).abs()

            # Sum of upper triangle (excluding diagonal)
            pairwise_sum = diffs.triu(diagonal=1).sum()

            # Number of pairs: m * (m - 1) / 2
            num_pairs = m * (m - 1) / 2
            mad = pairwise_sum / num_pairs

        return {self.name: mad.item()}
