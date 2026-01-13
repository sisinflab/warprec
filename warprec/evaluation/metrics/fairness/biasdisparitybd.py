# pylint: disable=arguments-differ, unused-argument, line-too-long, duplicate-code
from typing import Any

import torch
from torch import Tensor

from warprec.utils.registry import metric_registry
from warprec.evaluation.metrics.base_metric import TopKMetric, BaseMetric


@metric_registry.register("BiasDisparityBD")
class BiasDisparityBD(TopKMetric):
    """Bias Disparity (BD) metric.

    This metric measures the relative disparity in bias between the distribution of
    recommended items and the distribution of items in the training set,
    aggregated over user and item clusters.
    It is computed as the relative difference between BiasDisparityBR (bias in recommendations)
    and BiasDisparityBS (bias in the training set):

        BD(u, c) = (BR(u, c) - BS(u, c)) / BS(u, c)

    where:
        - u is a user cluster index,
        - c is an item cluster index,
        - BR(u, c) is the bias in recommendations for user cluster u and item cluster c,
        - BS(u, c) is the bias in the training set for user cluster u and item cluster c.

    A positive BD value indicates that the recommendation algorithm amplifies
    the bias compared to the training data, while a negative value indicates a reduction of bias.

    This metric internally uses the BiasDisparityBS and BiasDisparityBR metrics to compute its values.

    For further details, please refer to this `link <https://arxiv.org/pdf/1811.01461>`_.

    Attributes:
        bs_metric (BaseMetric): BiasDisparityBS metric instance.
        br_metric (BaseMetric): BiasDisparityBR metric instance.
        n_user_effective_clusters (int): The total number of unique user clusters.
        n_user_clusters (int): The total number of unique user clusters, including fallback cluster.
        n_item_effective_clusters (int): The total number of unique item clusters.
        n_item_clusters (int): The total number of unique item clusters, including fallback cluster.

    Args:
        k (int): Cutoff for top-k recommendations (used by BiasDisparityBR).
        num_items (int): Number of items in the training set.
        *args (Any): The argument list.
        user_cluster (Tensor): Lookup tensor of user clusters.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    bs_metric: BaseMetric
    br_metric: BaseMetric
    n_user_effective_clusters: int
    n_user_clusters: int
    n_item_effective_clusters: int
    n_item_clusters: int

    def __init__(
        self,
        k: int,
        num_items: int,
        *args: Any,
        user_cluster: Tensor = None,
        item_cluster: Tensor = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)

        # Initialize BiasDisparityBS and BiasDisparityBR
        self.bs_metric = metric_registry.get(
            "BiasDisparityBS",
            num_items=num_items,
            user_cluster=user_cluster,
            item_cluster=item_cluster,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )
        self.br_metric = metric_registry.get(
            "BiasDisparityBR",
            k=k,
            num_items=num_items,
            user_cluster=user_cluster,
            item_cluster=item_cluster,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )

        # Save cluster counts for output formatting
        self.n_user_clusters = self.bs_metric.n_user_clusters  # type: ignore[assignment]
        self.n_user_effective_clusters = self.bs_metric.n_user_effective_clusters  # type: ignore[assignment]
        self.n_item_clusters = self.bs_metric.n_item_clusters  # type: ignore[assignment]
        self.n_item_effective_clusters = self.bs_metric.n_item_effective_clusters  # type: ignore[assignment]

        # Update needed blocks to be the union of the blocks
        # of the two metrics
        self._REQUIRED_COMPONENTS = (
            self.bs_metric._REQUIRED_COMPONENTS | self.br_metric._REQUIRED_COMPONENTS
        )

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        # Update BiasDisparityBS (bias from training set)
        self.bs_metric.update(preds, **kwargs)

        # Update BiasDisparityBR (bias from top-k recommendations)
        self.br_metric.update(preds, **kwargs)

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
                br_val = br_results.get(br_key, 0.0)
                bs_tensor[uc, ic] = bs_val
                br_tensor[uc, ic] = br_val

        # Compute BD = (BR - BS) / BS, handle zero division safely
        safe_bs = bs_tensor.clone()
        safe_bs = safe_bs.clamp(min=1e-8)  # small epsilon to avoid division by zero
        bd_tensor = (br_tensor - bs_tensor) / safe_bs

        # Format results dictionary
        results = {}
        for uc in range(self.n_user_effective_clusters):
            for ic in range(self.n_item_effective_clusters):
                key = f"BiasDisparityBD_UC{uc + 1}_IC{ic + 1}"
                results[key] = bd_tensor[uc + 1, ic + 1].item()

        return results
