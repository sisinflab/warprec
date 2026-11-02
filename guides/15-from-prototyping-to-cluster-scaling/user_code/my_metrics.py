from typing import Any, Set

import torch
from torch import Tensor

from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("RBP")
class RBP(UserAverageTopKMetric):
    """Rank-biased precision (Moffat and Zobel, 2008), truncated at the cutoff.

    A hit at rank i (from 1) is worth (1 - p) * p ** (i - 1): a user who reads
    on from one item to the next with probability p sees it with that weight.

    Args:
        k (int): The cutoff.
        num_users (int): The number of users in the training set.
        *args (Any): Passed on to the base class.
        persistence (float): The probability p of reading on, in (0, 1).
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Everything else the evaluator passes, ignored here.

    Raises:
        ValueError: If persistence is not in (0, 1).
    """

    # The blocks compute_scores reads; the evaluator computes each once per batch.
    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    def __init__(
        self,
        k: int,
        num_users: int,
        *args: Any,
        persistence: float = 0.8,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, num_users, *args, dist_sync_on_step=dist_sync_on_step)
        if not 0 < persistence < 1:
            raise ValueError(f"persistence must be in (0, 1), got {persistence}.")
        self.persistence = persistence

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        """The RBP of every user in the batch.

        Args:
            preds (Tensor): The scores, unused here.
            target (Tensor): The binary relevance of every item, unused here.
            top_k_rel (Tensor): 1 where the item at each rank is relevant,
                shape [batch, k].
            **kwargs (Any): The other blocks.

        Returns:
            Tensor: One value per user, shape [batch].
        """
        ranks = torch.arange(top_k_rel.shape[1], device=top_k_rel.device)
        weights = (1 - self.persistence) * self.persistence**ranks
        return (top_k_rel * weights).sum(dim=1)

    @property
    def name(self) -> str:
        """RBP at the default persistence, RBP[p=...] at any other."""
        if self.persistence == 0.8:
            return "RBP"
        return f"RBP[p={self.persistence}]"
