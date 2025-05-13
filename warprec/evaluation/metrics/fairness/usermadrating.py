# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Optional, Dict

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("UserMADRating")
class UserMADRating(TopKMetric):
    """User MAD Rating (UserMADRating) metric.

    This metric measures the disparity in the average rating/score received by users
    across different user clusters, considering the average rating of their top-k
    recommended items. It computes the Mean Absolute Deviation (MAD) of the average
    per-user average top-k rating scores per user cluster. The MAD is computed as the mean
    of absolute differences between every pair of cluster-level averages.

    Formally, the metric is defined as:

        MAD = mean_{c_1, c_2} (|mean_avg_topk_score(c_1) - mean_avg_topk_score(c_2)|)

    where:
        - mean_avg_topk_score(c) is the average of the per-user average top-k prediction
          scores for all users in cluster c.
        - The per-user average top-k prediction score is the mean of the predicted scores
          of the top-k recommended items for that user.

    This metric is useful to detect disparities in the perceived quality (in terms of
    average predicted score) of recommendations across user clusters (e.g., based on
    demographics, behavior, etc.).

    The metric calculates the average predicted score of the top-k recommendations
    for each user and tracks the sum of these averages and the count of users
    per batch. These per-user statistics are then aggregated by cluster to compute
    cluster-level means of these average top-k scores, and their deviation is computed.

    For further details on the concept, please refer to this `link <https://dl.acm.org/doi/abs/10.1145/3269206.3271795>`_.

    Attributes:
        user_clusters (Tensor): Tensor mapping each user to an user cluster.
        user_counts (Tensor): Tensor of counts of batches/updates each user was seen in.
        user_gains (Tensor): Tensor of summed average top-k prediction scores per user across batches.
        n_users (int): Total number of users.

    Args:
        k (int): Cutoff for top-k recommendations.
        train_set (csr_matrix): Sparse matrix of training interactions (users x items).
        *args (Any): The argument list.
        user_cluster (Optional[Dict[int, int]]): Mapping from user IDs to user cluster IDs.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    user_clusters: Tensor
    user_counts: Tensor
    user_gains: Tensor
    n_users: int

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        *args: Any,
        user_cluster: Optional[Dict[int, int]] = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)

        # Prepare user-cluster mapping
        if user_cluster:
            unique_clusters = sorted(set(user_cluster.values()))
            self.n_user_clusters = len(unique_clusters)
            remap = {c: idx for idx, c in enumerate(unique_clusters)}
            uc = torch.zeros(train_set.shape[0], dtype=torch.long)
            for u, c in user_cluster.items():
                uc[u] = remap[c]
        else:
            self.n_user_clusters = 1
            uc = torch.zeros(train_set.shape[0], dtype=torch.long)
        self.register_buffer("user_clusters", uc)

        # Per-user accumulators
        self.add_state(
            "user_counts",
            torch.zeros(train_set.shape[0], dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "user_gains",
            torch.zeros(train_set.shape[0], dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor, start: int, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        topk = torch.topk(preds, self.k, dim=1, largest=True, sorted=True).values
        user_avg_scores = topk.mean(dim=1)  # [batch_size]

        # Global user indices for the current batch
        batch_size = preds.size(0)
        user_idx = torch.arange(start, start + batch_size, device=preds.device)

        self.user_gains.index_add_(0, user_idx, user_avg_scores)
        self.user_counts.index_add_(0, user_idx, torch.ones_like(user_avg_scores))

    def compute(self):
        """Computes the final metric value."""
        # Per-user mean nDCG
        mask = self.user_counts > 0
        v = torch.zeros_like(self.user_gains)
        v[mask] = self.user_gains[mask] / self.user_counts[mask]

        # Aggregate per cluster
        sum_cluster = torch.zeros(self.n_user_clusters, device=v.device)
        count_cluster = torch.zeros(
            self.n_user_clusters, dtype=torch.long, device=v.device
        )
        sum_cluster.scatter_add_(0, self.user_clusters, v)
        count_cluster.scatter_add_(0, self.user_clusters, mask.long())

        # Mean per cluster
        mean_cluster = torch.zeros_like(sum_cluster)
        nz = count_cluster > 0
        mean_cluster[nz] = sum_cluster[nz] / count_cluster[nz]

        # Pairwise absolute differences
        values = mean_cluster[nz]
        m = values.numel()
        if m < 2:
            mad = torch.tensor(0.0, device=values.device)
        else:
            # vectorized pairwise diffs
            i = values.unsqueeze(0)
            j = values.unsqueeze(1)
            diffs = (i - j).abs()
            # take upper triangle without diagonal
            pairwise = diffs.triu(diagonal=1)
            # mean of non-zero entries
            mad = pairwise.sum() / (m * (m - 1) / 2)

        return {self.name: mad.item()}

    def reset(self):
        """Resets the metric state."""
        self.user_counts.zero_()
        self.user_gains.zero_()
