from typing import Optional, Dict, Any
import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.utils.registry import metric_registry
from warprec.evaluation.base_metric import TopKMetric, BaseMetric


@metric_registry.register("BiasDisparityBD")
class BiasDisparityBD(TopKMetric):
    """Bias Disparity (BD) metric.

    This metric measures the relative disparity in bias between the distribution of recommended items
    and the distribution of items in the training set, aggregated over user and item clusters.
    It is computed as the relative difference between BiasDisparityBR (bias in recommendations)
    and BiasDisparityBS (bias in the training set):

        BD(u, c) = (BR(u, c) - BS(u, c)) / BS(u, c)

    where:
        - u is a user cluster index,
        - c is an item cluster index,
        - BR(u, c) is the bias in recommendations for user cluster u and item cluster c,
        - BS(u, c) is the bias in the training set for user cluster u and item cluster c.

    A positive BD value indicates that the recommendation algorithm amplifies the bias compared to the training data,
    while a negative value indicates a reduction of bias.

    This metric internally uses the BiasDisparityBS and BiasDisparityBR metrics to compute its values.

    For further details, please refer to this `link <https://arxiv.org/pdf/1811.01461>`_.

    Attributes:
        bs_metric (BaseMetric): BiasDisparityBS metric instance.
        br_metric (BaseMetric): BiasDisparityBR metric instance.
        n_user_clusters (int): Number of user clusters.
        n_item_clusters (int): Number of item clusters.

    Args:
        k (int): Cutoff for top-k recommendations (used by BiasDisparityBR).
        train_set (csr_matrix): Sparse matrix of training interactions (users x items).
        *args (Any): The argument list.
        user_cluster (Optional[Dict[int, int]]): Mapping from user IDs to user cluster IDs.
        item_cluster (Optional[Dict[int, int]]): Mapping from item IDs to item cluster IDs.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    bs_metric: BaseMetric
    br_metric: BaseMetric
    n_user_clusters: int
    n_item_clusters: int

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        *args: Any,
        user_cluster: Optional[Dict[int, int]] = None,
        item_cluster: Optional[Dict[int, int]] = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)

        # Initialize BiasDisparityBS and BiasDisparityBR
        self.bs_metric = metric_registry.get(
            "BiasDisparityBS",
            train_set=train_set,
            user_cluster=user_cluster,
            item_cluster=item_cluster,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )
        self.br_metric = metric_registry.get(
            "BiasDisparityBR",
            k=k,
            train_set=train_set,
            user_cluster=user_cluster,
            item_cluster=item_cluster,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )

        # Save cluster counts for output formatting
        self.n_user_clusters = self.bs_metric.n_user_clusters  # type: ignore[assignment]
        self.n_item_clusters = self.bs_metric.n_item_clusters  # type: ignore[assignment]

    def update(self, preds: Tensor, target: Tensor, start: int, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        # Update BiasDisparityBS (bias from training set)
        self.bs_metric.update(preds, target, start, **kwargs)

        # Update BiasDisparityBR (bias from top-k recommendations)
        self.br_metric.update(preds, target, start, **kwargs)

    def compute(self):
        """Computes the final metric value."""
        bs_results = self.bs_metric.compute()
        br_results = self.br_metric.compute()

        # Extract values into tensors for vectorized computation
        bs_tensor = torch.zeros((self.n_user_clusters, self.n_item_clusters))
        br_tensor = torch.zeros((self.n_user_clusters, self.n_item_clusters))

        # Fill tensors from dictionaries
        for uc in range(self.n_user_clusters):
            for ic in range(self.n_item_clusters):
                bs_key = f"BiasDisparityBS_UC{uc}_IC{ic}"
                br_key = f"BiasDisparityBR_UC{uc}_IC{ic}"
                bs_val = bs_results.get(bs_key, 1.0)  # Avoid division by zero fallback
                br_val = br_results.get(br_key, 1.0)
                bs_tensor[uc, ic] = bs_val
                br_tensor[uc, ic] = br_val

        # Compute BD = (BR - BS) / BS, handle zero division safely
        safe_bs = bs_tensor.clone()
        safe_bs = safe_bs.clamp(min=1e-8)  # small epsilon to avoid division by zero
        bd_tensor = (br_tensor - bs_tensor) / safe_bs

        # Format results dictionary
        results = {}
        for uc in range(self.n_user_clusters):
            for ic in range(self.n_item_clusters):
                key = f"BiasDisparityBD_UC{uc}_IC{ic}"
                results[key] = bd_tensor[uc, ic].item()

        return results

    def reset(self):
        """
        Reset the internal states of both BS and BR metrics.
        """
        self.bs_metric.reset()
        self.br_metric.reset()
