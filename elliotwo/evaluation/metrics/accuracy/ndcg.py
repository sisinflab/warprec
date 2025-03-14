# pylint: disable=arguments-differ, unused-argument
from typing import Any

import torch
from torch import Tensor
from elliotwo.evaluation.base_metric import TopKMetric
from elliotwo.utils.registry import metric_registry


@metric_registry.register("nDCG")
class nDCG(TopKMetric):
    """The nDCG@k metric is defined as the rapport of the DCG@k and the IDCG@k.

    The DCG@k represent the Discounted Cumulative Gain,
        which measures the gain of the items retrieved.

    The IDCG@k represent the Ideal Discounted Cumulative Gain,
        which measures the maximum gain possible
        obtainable by a perfect model.

    Attributes:
        ndcg (Tensor): The total value of ndcg per user.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff.
        dist_sync_on_step (bool): torchmetrics parameter.
        *args (Any): The argument list.
        **kwargs (Any): The keyword argument dictionary.
    """

    ndcg: Tensor
    users: Tensor

    def __init__(
        self, k: int, dist_sync_on_step: bool = False, *args: Any, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("ndcg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def dcg(self, rel: Tensor) -> Tensor:
        """The Discounted Cumulative Gain definition.

        Args:
            rel (Tensor): The relevance tensor.

        Returns:
            Tensor: The discounted tensor.
        """
        return (
            rel / torch.log2(torch.arange(2, rel.size(1) + 2, device=rel.device))
        ).sum(dim=1)

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        target = torch.where(target > 0, 2 ** (target + 1) - 1, target)
        top_k = torch.topk(preds, self.k, dim=1, largest=True, sorted=True).indices
        rel = torch.gather(target, 1, top_k).float()
        ideal_rel = torch.topk(target, self.k, dim=1, largest=True, sorted=True).values

        dcg_score = self.dcg(rel)
        idcg_score = self.dcg(ideal_rel).clamp(min=1e-10)

        self.ndcg += (dcg_score / idcg_score).nan_to_num(0).sum()
        self.users += target.shape[0]

    def compute(self):
        """Computes the final metric value."""
        return self.ndcg / self.users if self.users > 0 else torch.tensor(0.0)

    def reset(self):
        """Resets the metric state."""
        self.ndcg.zero_()
        self.users.zero_()
