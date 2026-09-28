from typing import Any, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import Tensor

from warprec.utils.registry import metric_registry
from warprec.evaluation.metrics.base_metric import BaseMetric


@metric_registry.register("BiasDisparityBS")
class BiasDisparityBS(BaseMetric):
    """BiasDisparityBS measures the bias each user cluster already carries in the training data.

    BS stands for Bias Source: the bias of the data a model learns from, which is the
    baseline the recommendation bias of BiasDisparityBR is read against. For a user
    cluster G and an item cluster C it is the share of G's training interactions that
    fall in C, divided by the share of the catalogue C occupies. A value of one means G
    consumes C exactly in proportion to its size.

    This is a property of the training split alone, so it is counted once when the
    metric is built and does not depend on the model, on the evaluation set or on K.

    Attributes:
        user_clusters (Tensor): Tensor mapping each user to a user cluster.
        item_clusters (Tensor): Tensor mapping each item to an item cluster.
        PC (Tensor): Global distribution of items across item clusters.
        category_sum (Tensor): Training interactions per user-item cluster pair.
        total_sum (Tensor): Training interactions per user cluster.

    Args:
        num_items (int): Number of items in the training set.
        user_cluster (Tensor): Lookup tensor of user clusters.
        item_cluster (Tensor): Lookup tensor of item clusters.
        train_matrix (Optional[csr_matrix]): The training interactions the bias is counted over.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ValueError: If the training interactions were not provided.
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
        train_matrix: Optional[csr_matrix] = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if train_matrix is None:
            raise ValueError(
                "BiasDisparityBS measures the bias of the training data, so the "
                "training interactions must be provided."
            )

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

        category_sum, total_sum = self._count_training_interactions(train_matrix)
        self.register_buffer("category_sum", category_sum)
        self.register_buffer("total_sum", total_sum)

    def _count_training_interactions(
        self, train_matrix: csr_matrix
    ) -> Tuple[Tensor, Tensor]:
        """Count the training interactions of every user-cluster / item-cluster pair.

        Args:
            train_matrix (csr_matrix): The training interactions.

        Returns:
            Tuple[Tensor, Tensor]: The per-pair counts and their per-user-cluster totals.
        """
        coo = train_matrix.tocoo()
        users = torch.from_numpy(coo.row.astype(np.int64))
        items = torch.from_numpy(coo.col.astype(np.int64))

        u_clusters = self.user_clusters[users]
        i_clusters = self.item_clusters[items]

        category_sum = torch.zeros(self.n_user_clusters, self.n_item_clusters)
        category_sum.index_put_(
            (u_clusters, i_clusters),
            torch.ones_like(u_clusters, dtype=torch.float),
            accumulate=True,
        )
        return category_sum, category_sum.sum(dim=1)

    def update(self, preds: Tensor, **kwargs: Any):
        """Accumulate nothing: the bias of the training data does not depend on a batch.

        Args:
            preds (Tensor): The predicted scores, unused.
            **kwargs (Any): The evaluation blocks, unused.
        """

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
