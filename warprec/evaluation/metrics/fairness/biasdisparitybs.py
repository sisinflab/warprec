from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.utils.registry import metric_registry
from warprec.evaluation.base_metric import BaseMetric


@metric_registry.register("BiasDisparityBS")
class BiasDisparityBS(BaseMetric):
    """BiasDisparityBS measures the disparity in recommendation bias across user and item clusters.

    This metric quantifies how the distribution of recommended items deviates from the global item
    distribution within each user cluster. It helps to identify whether certain user groups are
    disproportionately exposed to specific item categories compared to the overall item popularity.

    The metric is computed as:

        BiasDisparityBS(u, c) = (P_rec(u, c) / P_rec(u)) / P_global(c)

    where:
        - u is a user cluster index,
        - c is an item cluster index,
        - P_rec(u, c) is the proportion of recommended items from cluster c to users in cluster u,
        - P_rec(u) is the total number of recommendations to users in cluster u,
        - P_global(c) is the global proportion of items in cluster c.

    A value greater than 1 indicates over-recommendation of items from cluster c to user cluster u,
    while a value less than 1 indicates under-recommendation.

    For further details, please refer to this `link <https://arxiv.org/pdf/1811.01461>`_.

    Attributes:
        user_clusters (Tensor): Tensor mapping each user to a user cluster.
        n_user_effective_clusters (int): The total number of unique user clusters.
        n_user_clusters (int): The total number of unique user clusters, including fallback cluster.
        item_clusters (Tensor): Tensor mapping each item to an item cluster.
        n_item_effective_clusters (int): The total number of unique item clusters.
        n_item_clusters (int): The total number of unique item clusters, including fallback cluster.
        PC (Tensor): Global distribution of items across item clusters.
        category_sum (Tensor): Accumulated counts of positive interactions per user-item cluster pair.
        total_sum (Tensor): Accumulated counts of positive interactions per user cluster.

    Args:
        train_set (csr_matrix): Sparse matrix of training interactions (users x items).
        *args (Any): The argument list.
        user_cluster (Tensor): Lookup tensor of user clusters.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    user_clusters: Tensor
    n_user_effective_clusters: int
    n_user_clusters: int
    item_clusters: Tensor
    n_item_effective_clusters: int
    n_item_clusters: int
    PC: Tensor
    category_sum: Tensor
    total_sum: Tensor

    def __init__(
        self,
        train_set: csr_matrix,
        *args: Any,
        user_cluster: Tensor = None,
        item_cluster: Tensor = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.train_set = train_set

        self.register_buffer("user_clusters", user_cluster)
        self.n_user_effective_clusters = int(user_cluster.max().item())
        self.n_user_clusters = (
            self.n_user_effective_clusters + 1
        )  # Take into account the zero cluster

        self.register_buffer("item_clusters", item_cluster)
        self.n_item_effective_clusters = int(item_cluster.max().item())
        self.n_item_clusters = (
            self.n_item_effective_clusters + 1
        )  # Take into account the zero cluster

        # Global distribution of items
        pc = torch.bincount(item_cluster, minlength=self.n_item_clusters).float()
        pc = pc / float(self.train_set.shape[1])
        self.register_buffer("PC", pc)

        self.add_state(
            "category_sum",
            default=torch.zeros(self.n_user_clusters, self.n_item_clusters),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_sum", default=torch.zeros(self.n_user_clusters), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        start = kwargs.get("start", 0)
        target: Tensor = kwargs.get("ground", torch.zeros_like(preds))

        batch_size = preds.size(0)
        user_indices = torch.arange(start, start + batch_size, device=preds.device)

        # Find positive interactions in target
        user_batch_idx, item_idx = target.nonzero(as_tuple=True)

        # Map batch user indices to global user indices
        user_idx = user_indices[user_batch_idx]

        # Map to clusters
        grp = self.user_clusters[user_idx]
        cat = self.item_clusters[item_idx]

        # Accumulate counts
        self.category_sum.index_put_(
            (grp, cat), torch.ones_like(grp, dtype=torch.float), accumulate=True
        )
        self.total_sum.index_put_(
            (grp,), torch.ones_like(grp, dtype=torch.float), accumulate=True
        )

    def compute(self):
        """Computes the final metric value."""
        bias_src = (
            self.category_sum / self.total_sum.unsqueeze(1)
        ) / self.PC  # [n_user_clusters x n_item_clusters]
        results = {}
        for uc in range(self.n_user_effective_clusters):
            for ic in range(self.n_item_effective_clusters):
                key = f"{self.name}_UC{uc + 1}_IC{ic + 1}"
                results[key] = bias_src[uc + 1, ic + 1].item()
        return results

    def reset(self):
        """Resets the metric state."""
        self.category_sum.zero_()
        self.total_sum.zero_()
