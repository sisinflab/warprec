# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("HitRate")
class HitRate(TopKMetric):
    """The HitRate@k metric counts the number of users for which
        the model retrieved at least one item.

    The metric formula is defined as:
        HitRate@k = number_of_hits / n_users

    where:
        - number_of_hits is the number of users for which the model retrieved at least one item,

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

    HitRate@2 = (1 + 1) / 2 = 1.0

    For further details, please refer to this `link <https://en.wikipedia.org/wiki/Hit_rate>`_.

    Attributes:
        hits (Tensor): The number of hits in the top-k recommendations.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    hits: Tensor
    users: Tensor

    def __init__(
        self, k: int, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel: Tensor = kwargs.get(
            f"top_{self.k}_binary_relevance",
            self.top_k_relevance(preds, target, self.k),
        )

        self.hits += (top_k_rel.sum(dim=1) > 0).sum().float()

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final metric value."""
        hr = self.hits / self.users if self.users > 0 else torch.tensor(0.0)
        return {self.name: hr.item()}

    def reset(self):
        """Resets the metric state."""
        self.hits.zero_()
        self.users.zero_()
