# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("Recall")
class Recall(TopKMetric):
    """The Recall@k counts the number of item retrieve correctly,
        over the total number of relevant item in the ground truth.

    The metric formula is defined as:
        Recall@k = sum_{u=1}^{n_users} sum_{i=1}^{k} rel_{u,i} / (n_items * n_users)

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

    Recall@2 = (1 + 1) / (2 * 4) = 0.25

    For further details, please refer to this `link <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

    Attributes:
        correct (Tensor): The number of hits in the top-k recommendations.
        total_relevant (Tensor): The number of real hits in user transactions.

    Args:
        k (int): The cutoff.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    correct: Tensor
    total_relevant: Tensor

    def __init__(
        self, k: int, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "total_relevant", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        target = self.binary_relevance(target)
        top_k = torch.topk(preds, self.k, dim=1).indices
        rel = torch.gather(target, 1, top_k)
        self.correct += rel.sum().float()
        self.total_relevant += target.sum().float()

    def compute(self):
        """Computes the final metric value."""
        return (
            self.correct / self.total_relevant
            if self.total_relevant > 0
            else torch.tensor(0.0)
        )

    def reset(self):
        """Resets the metric state."""
        self.correct.zero_()
        self.total_relevant.zero_()
