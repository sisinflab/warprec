# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.base_metric import TopKMetric
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
        total_pop (Tensor): The total number of interaction for every
            item recommended.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff for recommendations.
        train_set (csr_matrix): The training interaction data.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_INDICES,
        MetricBlock.TOP_K_VALUES,
    }

    total_pop: Tensor
    users: Tensor

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.pop = self.compute_popularity(train_set).unsqueeze(0)
        self.add_state("total_pop", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
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

        self.total_pop += (rel * batch_pop).sum()

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final metric value."""
        arp = (
            self.total_pop / (self.users * self.k)
            if self.users > 0
            else torch.tensor(0.0)
        )
        return {self.name: arp.item()}

    def reset(self):
        """Resets the metric state."""
        self.total_pop.zero_()
        self.users.zero_()
