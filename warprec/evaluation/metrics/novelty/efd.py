from typing import Any

import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("EFD")
class EFD(UserAverageTopKMetric):
    """Expected Free Discovery at K metric.

    This metric measures the recommender system's ability to suggest items
    that the user has not already seen (i.e., not present in the training set).

    Attributes:
        novelty_profile (Tensor): The item novelty lookup tensor.
        relevance (str): The type of relevance to use for computation.

    Args:
        k (int): The cutoff for recommendations.
        num_users (int): Number of users in the training set.
        item_interactions (Tensor): The counts for item interactions in training set.
        dist_sync_on_step (bool): Torchmetrics parameter.
        relevance (str): The type of relevance to use for computation.
        **kwargs (Any): Additional keyword arguments.
    """

    novelty_profile: Tensor
    relevance: str

    def __init__(
        self,
        k: int,
        num_users: int,
        item_interactions: Tensor,
        dist_sync_on_step: bool = False,
        relevance: str = "binary",
        **kwargs: Any,
    ):
        super().__init__(k=k, num_users=num_users, dist_sync_on_step=dist_sync_on_step)
        self.relevance = relevance

        # Add novelty profile as buffer
        self.register_buffer(
            "novelty_profile",
            self.compute_novelty_profile(
                item_interactions, num_users, log_discount=True
            ),
        )

        # Check for requirements
        self._REQUIRED_COMPONENTS = (
            {MetricBlock.DISCOUNTED_RELEVANCE, MetricBlock.TOP_K_DISCOUNTED_RELEVANCE}
            if relevance == "discounted"
            else {MetricBlock.BINARY_RELEVANCE, MetricBlock.TOP_K_BINARY_RELEVANCE}
        )
        self._REQUIRED_COMPONENTS.add(MetricBlock.VALID_USERS)
        self._REQUIRED_COMPONENTS.add(MetricBlock.TOP_K_INDICES)

    def unpack_inputs(self, preds: Tensor, **kwargs: Any):
        users = kwargs.get("valid_users")

        # Handle relevance types
        if self.relevance == "discounted":
            target = kwargs.get("discounted_relevance")
            top_k_rel = kwargs.get(f"top_{self.k}_discounted_relevance")
            return target, users, top_k_rel
        target = kwargs.get("binary_relevance")
        top_k_rel = kwargs.get(f"top_{self.k}_binary_relevance")
        return target, users, top_k_rel

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        top_k_indices = kwargs.get(f"top_{self.k}_indices")
        item_indices = kwargs.get("item_indices")

        # Retrieve Novelty for Top-K items
        if item_indices is not None:
            batch_novelty = self.novelty_profile[0, item_indices]
            novelty = torch.gather(batch_novelty, 1, top_k_indices)
        else:
            novelty = self.novelty_profile[0, top_k_indices]

        # Compute DCG(rel * novelty)
        gain = top_k_rel * novelty
        dcg_val = self.dcg(gain)

        # Normalize by Discounted Sum (IDCG-like factor)
        return dcg_val / self.discounted_sum(self.k)

    @property
    def name(self):
        """The name of the metric."""
        if self.relevance == "binary":
            return self.__class__.__name__
        return f"EFD[{self.relevance}]"
