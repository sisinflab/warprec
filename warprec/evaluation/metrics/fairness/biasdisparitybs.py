from typing import Any

import torch
from torch import Tensor

from warprec.utils.registry import metric_registry
from warprec.evaluation.metrics.base_metric import BaseMetric


@metric_registry.register("BiasDisparityBS")
class BiasDisparityBS(BaseMetric):
    """BiasDisparityBS measures the disparity in recommendation bias across user and item clusters.

    This metric quantifies how the distribution of recommended items deviates from the global item
    distribution within each user cluster. It helps to identify whether certain user groups are
    disproportionately exposed to specific item categories compared to the overall item popularity.

    The metric is computed as:

        BiasDisparityBS(u, c) = P_train(u, c) / P_global(c)

    where:
        - u is a user cluster index,
        - c is an item cluster index,
        - P_train(u, c) is the proportion of positive interactions from user cluster u with items in cluster c within the training set,
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
        num_items (int): Number of items in the training set.
        user_cluster (Tensor): Lookup tensor of user clusters.
        item_cluster (Tensor): Lookup tensor of item clusters.
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
        num_items: int,
        user_cluster: Tensor,
        item_cluster: Tensor,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Register static buffers
        self.register_buffer("user_clusters", user_cluster)
        self.register_buffer("item_clusters", item_cluster)

        self.n_user_effective_clusters = int(user_cluster.max().item())
        self.n_user_clusters = self.n_user_effective_clusters + 1

        self.n_item_effective_clusters = int(item_cluster.max().item())
        self.n_item_clusters = self.n_item_effective_clusters + 1

        # Global distribution of items (P_global)
        pc = torch.bincount(item_cluster, minlength=self.n_item_clusters).float()
        pc = pc / float(num_items)
        self.register_buffer("PC", pc)

        # Accumulators
        self.add_state(
            "category_sum",
            default=torch.zeros(self.n_user_clusters, self.n_item_clusters),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_sum", default=torch.zeros(self.n_user_clusters), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        # Retrieve Ground Truth (Binary or Raw)
        target = kwargs.get("ground")

        # Find positive interactions in the batch
        user_idx_local, item_idx_local = target.nonzero(as_tuple=True)

        if user_idx_local.numel() == 0:
            return

        # Map User Indices: Batch -> Global
        user_idx_global = user_indices[user_idx_local]

        # Map Item Indices: Local -> Global
        item_indices = kwargs.get("item_indices")
        if item_indices is not None:
            item_idx_global = item_indices[user_idx_local, item_idx_local]
        else:
            item_idx_global = item_idx_local

        # Get Clusters
        u_clusters = self.user_clusters[user_idx_global]
        i_clusters = self.item_clusters[item_idx_global]

        # Accumulate
        # We use index_put_ with accumulate=True for efficient scatter add
        self.category_sum.index_put_(
            (u_clusters, i_clusters),
            torch.ones_like(u_clusters, dtype=torch.float),
            accumulate=True,
        )
        self.total_sum.index_put_(
            (u_clusters,),
            torch.ones_like(u_clusters, dtype=torch.float),
            accumulate=True,
        )

    def compute(self):
        # P_train(u, c) / P_global(c)
        # Avoid division by zero for total_sum
        safe_total = self.total_sum.unsqueeze(1).clamp(min=1.0)

        bias_src = (self.category_sum / safe_total) / self.PC.unsqueeze(0)

        results = {}
        for uc in range(self.n_user_effective_clusters):
            for ic in range(self.n_item_effective_clusters):
                # +1 because cluster 0 is usually padding/unknown
                key = f"{self.name}_UC{uc + 1}_IC{ic + 1}"
                results[key] = bias_src[uc + 1, ic + 1].item()
        return results
