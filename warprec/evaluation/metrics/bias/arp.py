# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("ARP")
class ARP(TopKMetric):
    """ARP (Average Recommendation Popularity) is a metric that evaluates
    the average popularity of the top-k recommendations.

    The metric formula is defined as:
        ARP = sum(long_hits) / (users * k)

    where:
        -long_hits are the number of recommendation in the long tail.

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

    then we extract the relevance (original score) for that user in that column but maintaining the original dimensions:
           REL
    +---+---+---+---+
    | 0 | 0 | 1 | 0 |
    | 0 | 0 | 0 | 1 |
    +---+---+---+---+

    then we compute the item counts using the column indices:
         COUNTS
    +---+---+---+---+
    | 0 | 0 | 1 | 1 |
    +---+---+---+---+

    Finally we multiply the relevance by the item counts (in this case we have only ones, so the result is the same):
           POP
    +---+---+---+---+
    | 0 | 0 | 1 | 0 |
    | 0 | 0 | 0 | 1 |
    +---+---+---+---+

    For further details, please refer to this `paper <https://arxiv.org/abs/1901.07555>`_.

    Attributes:
        retrieved_pop (Tensor): The number of interaction for every
            item recommended.
        users (Tensor): The number of users evaluated.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.

    Args:
        k (int): The cutoff for recommendations.
        num_users (int): Number of users in the training set.
        item_interactions (Tensor): The counts for item interactions in training set.
        *args (Any): The argument list.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_INDICES,
        MetricBlock.TOP_K_VALUES,
    }
    _CAN_COMPUTE_PER_USER: bool = True

    retrieved_pop: Tensor
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
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.pop = self.compute_popularity(item_interactions)
        self.compute_per_user = compute_per_user

        if self.compute_per_user:
            self.add_state(
                "retrieved_pop", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )  # Initialize a tensor to store metric value for each user
        else:
            self.add_state(
                "retrieved_pop", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )  # Initialize a scalar to store global value
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_values: Tensor = kwargs.get(
            f"top_{self.k}_values", self.top_k_values_indices(preds, self.k)[0]
        )
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )

        rel = torch.zeros_like(preds)
        rel.scatter_(
            dim=1, index=top_k_indices, src=top_k_values
        )  # [batch_size x items]
        rel = rel * target  # [batch_size x items]
        rel[rel > 0] = 1

        batch_pop = self.pop.repeat(target.shape[0], 1)  # [batch_size x items]

        if self.compute_per_user:
            self.retrieved_pop.index_add_(0, user_indices, (rel * batch_pop).sum(dim=1))
        else:
            self.retrieved_pop += (rel * batch_pop).sum()

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final metric value."""
        if self.compute_per_user:
            arp = self.retrieved_pop / self.k
        else:
            arp = (
                self.retrieved_pop / (self.users * self.k)
                if self.users > 0
                else torch.tensor(0.0)
            ).item()
        return {self.name: arp}
