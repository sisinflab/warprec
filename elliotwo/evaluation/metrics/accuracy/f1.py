# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("F1")
class F1(TopKMetric):
    """The F1@k metric combines precision and recall at k, providing a harmonic mean
    between the two to evaluate the relevance of the top-k recommended items.

    This implementation follows the standard F1 formula:
        F1@k = (1 + beta^2) * (Precision@k * Recall@k) / (beta^2 * Precision@k + Recall@k)

    where:
        Precision@k = sum_{u=1}^{n_users} sum_{i=1}^{k} rel_{u,i} / (k * n_users)
        Recall@k = sum_{u=1}^{n_users} sum_{i=1}^{k} rel_{u,i} / (n_items * n_users)

    For the matrix computation of the metric, please refer to the Precision@k and Recall@k classes.

    For further details, please refer to this `book <https://link.springer.com/chapter/10.1007/978-1-4899-7637-6_8>`_
    and this `link <https://en.wikipedia.org/wiki/Precision_and_recall>`_.

    Attributes:
        correct (Tensor): The number of hits in the top-k recommendations.
        total_relevant (Tensor): The number of real hits in user transactions.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The number of top recommendations to consider (cutoff).
        *args (Any): Additional arguments to pass to the parent class.
        beta (float): The weight of recall in the harmonic mean. Default is 1.0.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    correct: Tensor
    total_relevant: Tensor
    users: Tensor

    def __init__(
        self,
        k: int,
        *args: Any,
        beta: float = 1.0,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.beta = beta
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "total_relevant", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        target = self.binary_relevance(target)
        top_k = torch.topk(preds, self.k, dim=1).indices
        rel = torch.gather(target, 1, top_k)

        # Update precision and recall
        self.correct += rel.sum().float()
        self.total_relevant += target.sum().float()

        # Count only users with at least one interaction
        self.users += (target > 0).any(dim=1).sum().item()

    def compute(self):
        """Computes the F1 score using precision and recall."""
        precision = (
            self.correct / (self.users * self.k)
            if self.users > 0
            else torch.tensor(0.0)
        )
        recall = (
            self.correct / self.total_relevant
            if self.total_relevant > 0
            else torch.tensor(0.0)
        )

        f1_score = (
            (1 + self.beta**2)
            * (precision * recall)
            / (self.beta**2 * precision + recall)
            if precision + recall > 0
            else torch.tensor(0.0)
        )
        return f1_score

    def reset(self):
        """Resets the metric state."""
        self.correct.zero_()
        self.total_relevant.zero_()
        self.users.zero_()
