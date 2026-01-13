# pylint: disable=arguments-differ, unused-argument, line-too-long, duplicate-code
from typing import Any

import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("EFD")
class EFD(TopKMetric):
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
        efd (Tensor): The EFD value for every user.
        users (Tensor): Number of users evaluated.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.

    Args:
        k (int): The cutoff for recommendations.
        num_users (int): Number of users in the training set.
        item_interactions (Tensor): The counts for item interactions in training set.
        *args (Any): Additional arguments.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        dist_sync_on_step (bool): Torchmetrics parameter.
        relevance (str): The type of relevance to use for computation.
        **kwargs (Any): Additional keyword arguments.
    """

    _CAN_COMPUTE_PER_USER: bool = True

    novelty_profile: Tensor
    efd: Tensor
    users: Tensor
    compute_per_user: bool

    def __init__(
        self,
        k: int,
        num_users: int,
        item_interactions: Tensor,
        *args: Any,
        compute_per_user: bool = False,
        dist_sync_on_step: bool = False,
        relevance: str = "binary",
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.relevance = relevance
        self.compute_per_user = compute_per_user

        if self.compute_per_user:
            self.add_state(
                "efd", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )  # Initialize a tensor to store metric value for each user
        else:
            self.add_state(
                "efd", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )  # Initialize a scalar to store global value
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

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

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs):
        """Updates the metric state with a new batch of predictions."""
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )
        target: Tensor = None
        top_k_rel: Tensor = None
        if self.relevance == "discounted":
            target = kwargs.get("discounted_relevance", torch.zeros_like(preds))
            top_k_rel = kwargs.get(
                f"top_{self.k}_discounted_relevance",
                self.top_k_relevance(preds, target, self.k),
            )
        else:
            target = kwargs.get("binary_relevance", torch.zeros_like(preds))
            top_k_rel = kwargs.get(
                f"top_{self.k}_binary_relevance",
                self.top_k_relevance(preds, target, self.k),
            )
        users = kwargs.get("valid_users", self.valid_users(target))

        # Extract novelty values
        batch_novelty = self.novelty_profile.repeat(
            target.shape[0], 1
        )  # [batch_size x items]
        novelty = torch.gather(batch_novelty, 1, top_k_indices)  # [batch_size x top_k]

        if self.compute_per_user:
            self.efd.index_add_(0, user_indices, self.dcg(top_k_rel * novelty))
        else:
            self.efd += self.dcg(top_k_rel * novelty).sum()

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final value of the metric."""
        if self.compute_per_user:
            efd = self.efd / self.discounted_sum(self.k)
        else:
            efd = (
                self.efd / (self.users * self.discounted_sum(self.k))
                if self.users > 0
                else torch.tensor(0.0)
            ).item()
        return {self.name: efd}

    @property
    def name(self):
        """The name of the metric."""
        if self.relevance == "binary":
            return self.__class__.__name__
        return f"EFD[{self.relevance}]"
