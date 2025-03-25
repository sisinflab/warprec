# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("Precision")
class Precision(TopKMetric):
    """The Precision@k counts the number of item retrieved correctly,
        over the maximum number of possible retrieve items.

    The metric formula is defined as:
        Precision@k = sum_{u=1}^{n_users} sum_{i=1}^{k} rel_{u,i} / (k * n_users)

    where:
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

    Precision@2 = (1 + 1) / (2 * 2) = 0.5

    For further details, please refer to this `link <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

    Attributes:
        correct (Tensor): The number of hits in the top-k recommendations.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    correct: Tensor
    users: Tensor

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        target = target.clone()
        target[target > 0] = 1
        top_k = torch.topk(preds, self.k, dim=1).indices
        rel = torch.gather(target, 1, top_k)
        self.correct += rel.sum().float()
        self.users += target.shape[0]

    def compute(self):
        """Computes the final metric value."""
        return (
            self.correct / (self.users * self.k)
            if self.users > 0
            else torch.tensor(0.0)
        )

    def reset(self):
        """Resets the metric state."""
        self.correct.zero_()
        self.users.zero_()
