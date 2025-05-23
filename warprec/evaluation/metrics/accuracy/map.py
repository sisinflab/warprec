# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from warprec.evaluation.base_metric import TopKMetric
from warprec.utils.registry import metric_registry


@metric_registry.register("MAP")
class MAP(TopKMetric):
    """Mean Average Precision (MAP) at K.

    MAP@K calculates the mean of the Average Precision for all users.
    It considers the position of relevant items in the recommendation list.

    The metric formula is defined as:
        MAP@K = sum_{u=1}^{n_users} sum_{i=1}^{k} (P@i * rel_{u,i}) / n_users

    where:
        - P@i is the precision at i-th position.
        - rel_{u,i} is the relevance of the i-th item for user u.

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

    The precision at i-th position is calculated as the sum of the relevant items
    divided by the position:
    PRECISION
    +---+----+
    | 0 | .5 |
    | 1 | 0  |
    +---+----+

    the normalization is the minimum between the number of relevant items and k:
    NORMALIZATION
    +---+---+
    | 2 | 2 |
    +---+---+

    MAP@2 = 1 / 2 + 0.5 / 2 = 0.75

    For further details, please refer to this `link <https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms>`_.

    Attributes:
        ap_sum (Tensor): The average precision tensor.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The recommendation cutoff.
        *args (Any): Additional arguments to pass to the parent class.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    ap_sum: Tensor
    users: Tensor

    def __init__(
        self, k: int, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("ap_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the MAP metric state with a batch of predictions."""
        target = kwargs.get("binary_relevance", torch.zeros_like(preds))

        top_k = torch.topk(preds, self.k, dim=1).indices
        rel = torch.gather(target, 1, top_k)

        precision_at_i = rel.cumsum(dim=1) / torch.arange(
            1, self.k + 1, device=rel.device
        )  # [batch_size, k]
        normalization = torch.minimum(
            target.sum(dim=1),
            torch.tensor(self.k, dtype=target.dtype, device=target.device),
        )  # [batch_size]
        ap = (precision_at_i * rel).sum(dim=1) / normalization  # [batch_size]

        self.ap_sum += ap.sum()

        # Count only users with at least one interaction
        self.users += (target > 0).any(dim=1).sum().item()

    def compute(self):
        """Computes the final MAP@K value."""
        map = self.ap_sum / self.users if self.users > 0 else torch.tensor(0.0)
        return {self.name: map.item()}
