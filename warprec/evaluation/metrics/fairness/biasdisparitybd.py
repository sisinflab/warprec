from typing import Any

from torch import Tensor

from warprec.utils.registry import metric_registry
from warprec.evaluation.metrics.base_metric import TopKMetric


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

    Args:
        k (int): Cutoff for top-k recommendations (used by BiasDisparityBR).
        num_items (int): Number of items in the training set.
        user_cluster (Tensor): Lookup tensor of user clusters.
        item_cluster (Tensor): Lookup tensor of item clusters.
        dist_sync_on_step (bool): Whether to synchronize metric state across distributed processes.
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(
        self,
        k: int,
        num_items: int,
        user_cluster: Tensor,
        item_cluster: Tensor,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k=k, dist_sync_on_step=dist_sync_on_step)

        # Instantiate sub-metrics
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

        # Union of required components
        self._REQUIRED_COMPONENTS = (
            self.bs_metric._REQUIRED_COMPONENTS | self.br_metric._REQUIRED_COMPONENTS
        )

        # Metadata for formatting
        self.n_user_effective_clusters = self.bs_metric.n_user_effective_clusters
        self.n_item_effective_clusters = self.bs_metric.n_item_effective_clusters

    def update(self, preds: Tensor, **kwargs: Any):
        # Update sub-metrics inner states
        self.bs_metric.update(preds, **kwargs)
        self.br_metric.update(preds, **kwargs)

    def compute(self):
        # Compute BS Tensor
        bs_safe_total = self.bs_metric.total_sum.unsqueeze(1).clamp(min=1.0)
        bs_tensor = (
            self.bs_metric.category_sum / bs_safe_total
        ) / self.bs_metric.PC.unsqueeze(0)

        # Compute BR Tensor
        br_safe_total = self.br_metric.total_sum.unsqueeze(1).clamp(min=1.0)
        br_tensor = (
            self.br_metric.category_sum / br_safe_total
        ) / self.br_metric.PC.unsqueeze(0)

        # Compute BD handling division by zero
        bd_tensor = ((br_tensor - bs_tensor) / bs_tensor).nan_to_num(0.0)

        results = {}

        # Format output per cluster combination
        for uc in range(self.n_user_effective_clusters):
            for ic in range(self.n_item_effective_clusters):
                key = f"{self.name}_UC{uc + 1}_IC{ic + 1}"
                results[key] = bd_tensor[uc + 1, ic + 1].item()

        return results
