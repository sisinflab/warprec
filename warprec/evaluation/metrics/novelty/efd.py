from typing import Any

import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("EFD")
class EFD(UserAverageTopKMetric):
    """
    Expected Free Discovery at K metric.

    This metric measures the recommender system's ability to suggest items
    that the user has not already seen (i.e., not present in the training set).

    The metric formula is defines as:
        EFD = sum(DCG(rel * novelty)) / (users * discounted_sum)

    where:
        - DCG is the discounted cumulative gain.
        - rel is the relevance of the items.
        - novelty is the novelty of the items.
        - users is the number of users evaluated.
        - discounted_sum is the sum of the discounted values for the top-k items.

    Matrix computation of the metric:
        PREDS                   TARGETS
    +---+---+---+---+       +---+---+---+---+
    | 8 | 2 | 7 | 2 |       | 1 | 0 | 1 | 0 |
    | 5 | 4 | 3 | 9 |       | 0 | 0 | 1 | 1 |
    +---+---+---+---+       +---+---+---+---+

    We extract the top-k predictions and get their column index. Let's assume k=2:
      TOP-K
    +---+---+
    | 0 | 2 |
    | 3 | 0 |
    +---+---+

    then we extract the relevance (original score) for that user in that column:
       REL
    +---+---+
    | 0 | 1 |
    | 1 | 0 |
    +---+---+

    The discounted novelty score of an item is computed as:

    DiscountedNovelty_i = -log_2(interactions_i / users)

    where:
        -interactions_i is the number of times the item i has been interacted with.
        -users is the total number of users.

    The novelty is expressed as a tensor of length equal to the number of items. This is repeated
        for each user in the current batch.

    The discounted sum is computed as (for k=2):

    DiscountedSum@2 = 1/log_2(2) + 1/log_2(3) = 1.63

    For further details, please refer to this `link <https://dl.acm.org/doi/abs/10.1145/2043932.2043955>`_.

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
