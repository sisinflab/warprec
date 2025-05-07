from typing import Dict, Optional, Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.utils.registry import metric_registry
from warprec.evaluation.base_metric import BaseMetric


@metric_registry.register("BiasDisparityBS")
class BiasDisparityBS(BaseMetric):
    """
    BiasDisparityBS measures the disparity in recommendation bias across user and item clusters.

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
        item_clusters (Tensor): Tensor mapping each item to an item cluster.
        PC (Tensor): Global distribution of items across item clusters.
        category_sum (Tensor): Accumulated counts of positive interactions per user-item cluster pair.
        total_sum (Tensor): Accumulated counts of positive interactions per user cluster.

    Args:
        train_set (csr_matrix): Sparse matrix of training interactions (users x items).
        *args (Any): The argument list.
        user_cluster (Optional[Dict[int, int]]): Mapping from user IDs to user cluster IDs.
        item_cluster (Optional[Dict[int, int]]): Mapping from item IDs to item cluster IDs.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    user_clusters: Tensor
    item_clusters: Tensor
    PC: Tensor
    category_sum: Tensor
    total_sum: Tensor

    def __init__(
        self,
        train_set: csr_matrix,
        *args: Any,
        user_cluster: Optional[Dict[int, int]] = None,
        item_cluster: Optional[Dict[int, int]] = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.train_set = train_set

        # Handle user clusters
        if user_cluster:
            unique_user_clusters = sorted(set(user_cluster.values()))
            self.n_user_clusters = len(unique_user_clusters)
            user_cluster_remap = {
                cid: idx for idx, cid in enumerate(unique_user_clusters)
            }  # Use appropriate indexes for clusters
            uc = torch.zeros(self.train_set.shape[0], dtype=torch.long)
            for u, g in user_cluster.items():
                uc[u] = user_cluster_remap[g]
        else:
            self.n_user_clusters = 1
            uc = torch.zeros(self.train_set.shape[0], dtype=torch.long)
        self.register_buffer("user_clusters", uc)

        # Handle item clusters
        if item_cluster:
            unique_item_clusters = sorted(set(item_cluster.values()))
            self.n_item_clusters = len(unique_item_clusters)
            item_cluster_remap = {
                cid: idx for idx, cid in enumerate(unique_item_clusters)
            }  # Use appropriate indexes for clusters
            ic = torch.zeros(self.train_set.shape[1], dtype=torch.long)
            for i, c in item_cluster.items():
                ic[i] = item_cluster_remap[c]
        else:
            self.n_item_clusters = 1
            ic = torch.zeros(self.train_set.shape[1], dtype=torch.long)
        self.register_buffer("item_clusters", ic)

        # Global distribution of items
        pc = torch.bincount(ic, minlength=self.n_item_clusters).float()
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

    def update(self, preds: Tensor, target: Tensor, start: int, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
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
        for uc in range(self.n_user_clusters):
            for ic in range(self.n_item_clusters):
                key = f"{self.name}_UC{uc}_IC{ic}"
                results[key] = bias_src[uc, ic].item()
        return results

    def reset(self):
        """Resets the metric state."""
        self.category_sum.zero_()
        self.total_sum.zero_()
