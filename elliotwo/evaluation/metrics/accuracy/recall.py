# pylint: disable=arguments-differ
from typing import Any

import torch
from torch import Tensor
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("Recall")
class Recall(TopKMetric):
    """The Recall@k counts the number of item retrieve correctly,
        over the total number of relevant item in the ground truth.

    Attributes:
        correct (Tensor): The number of hits in the top-k recommendations.
        real_correct (Tensor): The number of real hits in user transactions.

    Args:
        k (int): The cutoff.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    correct: Tensor
    real_correct: Tensor

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("real_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        target = target.clone()
        target[target > 0] = 1
        top_k = torch.topk(preds, self.k, dim=1).indices
        rel = torch.gather(target, 1, top_k)
        self.correct += rel.sum().float()
        self.real_correct += target.sum().float()

    def compute(self):
        """Computes the final metric value."""
        return (
            self.correct / self.real_correct
            if self.real_correct > 0
            else torch.tensor(0.0)
        )

    def reset(self):
        """Resets the metric state."""
        self.correct.zero_()
        self.real_correct.zero_()
