# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("MRR")
class MRR(TopKMetric):
    """Mean Reciprocal Rank (MRR) at K.

    MRR measures the position of the first relevant item in the recommendation list.

    The metric formula is defined as:
        MRR@K = sum_{u=1}^{n_users} (1 / rank_u) / n_users

    where:
        - rank_u is the position of the first relevant item in the recommendation list.

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

    The reciprocal rank is calculated as the inverse of the rank of the first relevant item:
    RECIPROCAL RANK
    +----+---+
    | .5 | 1 |
    +----+---+

    MRR@2 = (0.5 + 1) / 2 = 0.75

    Attributes:
        reciprocal_rank_sum (Tensor): The reciprocal rank sum tensor.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The recommendation cutoff.
        *args (Any): Additional arguments to pass to the parent class.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    reciprocal_rank_sum: Tensor
    users: Tensor

    def __init__(
        self, k: int, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state(
            "reciprocal_rank_sum", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the MRR metric state with a batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel: Tensor = kwargs.get(
            f"top_{self.k}_binary_relevance",
            self.top_k_relevance(preds, target, self.k),
        )

        # Find the first relevant item's rank
        reciprocal_ranks = (top_k_rel.argmax(dim=1) + 1).float().reciprocal()
        reciprocal_ranks[top_k_rel.sum(dim=1) == 0] = 0  # Assign 0 if no relevant items

        self.reciprocal_rank_sum += reciprocal_ranks.sum()

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final MRR@K value."""
        mrr = (
            self.reciprocal_rank_sum / self.users
            if self.users > 0
            else torch.tensor(0.0)
        )
        return {self.name: mrr.item()}
