# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.base_metric import TopKMetric
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
        efd (Tensor): The EFD value for every user.
        users (Tensor): Number of users evaluated.

    Args:
        k (int): The cutoff for recommendations.
        train_set (csr_matrix): The training interaction data.
        *args (Any): Additional arguments.
        dist_sync_on_step (bool): Torchmetrics parameter.
        relevance (str): The type of relevance to use for computation.
        **kwargs (Any): Additional keyword arguments.
    """

    efd: Tensor
    users: Tensor

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        *args: Any,
        dist_sync_on_step: bool = False,
        relevance: str = "binary",
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.novelty_profile = self.compute_novelty_profile(
            train_set, log_discount=True
        ).unsqueeze(0)
        self.relevance = relevance
        self.add_state("efd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Check for requirements
        self._REQUIRED_COMPONENTS = (
            {MetricBlock.DISCOUNTED_RELEVANCE, MetricBlock.TOP_K_DISCOUNTED_RELEVANCE}
            if relevance == "discounted"
            else {MetricBlock.BINARY_RELEVANCE, MetricBlock.TOP_K_BINARY_RELEVANCE}
        )
        self._REQUIRED_COMPONENTS.add(MetricBlock.VALID_USERS)
        self._REQUIRED_COMPONENTS.add(MetricBlock.TOP_K_INDICES)

    def update(self, preds: Tensor, **kwargs):
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

        # Update
        self.efd += self.dcg(top_k_rel * novelty).sum()

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final value of the metric."""
        score = (
            self.efd / (self.users * self.discounted_sum(self.k))
            if self.users > 0
            else torch.tensor(0.0)
        )
        return {self.name: score.item()}

    @property
    def name(self):
        """The name of the metric."""
        if self.relevance == "binary":
            return self.__class__.__name__
        return f"EFD[{self.relevance}]"
