from typing import Any, Optional, Dict

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("UserMADRanking")
class UserMADRanking(TopKMetric):
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
        user_counts (Tensor): Tensor of counts of user recommended.
        user_gains (Tensor): Tensor of gains of user recommended.

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
        # Apply discounted relevance (2^rel - 1) to target
        discounted = self.discounted_relevance(target)

        # Top-k item indices by prediction
        topk = torch.topk(preds, self.k, dim=1, largest=True, sorted=True)
        topk_idx, _ = topk.indices, topk.values  # indices: (batch, k)

        # Gather relevance at top-k
        rel = discounted.gather(1, topk_idx)

        # Compute ideal (best) relevance for IDCG
        ideal_rel = torch.topk(
            discounted, self.k, dim=1, largest=True, sorted=True
        ).values

        # Compute DCG and IDCG per user
        dcg_score = self.dcg(rel)
        idcg_score = self.dcg(ideal_rel).clamp(min=1e-10)

        # nDCG per user
        ndcg_scores = (dcg_score / idcg_score).nan_to_num(0)

        # Global user indices
        batch_size = preds.size(0)
        user_idx = torch.arange(start, start + batch_size, device=preds.device)

        # Accumulate per-user values
        self.user_gains.index_add_(0, user_idx, ndcg_scores)
        self.user_counts.index_add_(0, user_idx, torch.ones_like(ndcg_scores))

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
