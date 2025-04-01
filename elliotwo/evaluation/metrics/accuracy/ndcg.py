# pylint: disable=arguments-differ, unused-argument, line-too-long
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

    The metric formula is defined as:
        nDCG@k = DCG@k / IDCG@k

    where:
        - DCG@k = sum_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)
        - IDCG@k = sum_{i=1}^{k} (2^ideal_rel_i - 1) / log2(i + 1)

    Matrix computation of the metric:
        PREDS                   TARGETS
    +---+---+---+---+       +---+---+---+---+
    | 8 | 2 | 7 | 2 |       | 3 | 0 | 1 | 0 |
    | 5 | 4 | 3 | 9 |       | 0 | 0 | 2 | 5 |
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
    | 5 | 0 |
    +---+---+

    The relevance considered for the nDCG score is discounted as 2^(rel + 1) - 1.
       REL
    +----+---+
    | 0  | 3 |
    | 63 | 0 |
    +----+---+

    The ideal relevance is computed by taking the top-k items from the target tensor:
    IDEAL REL
    +----+---+
    | 15 | 3 |
    | 63 | 7 |
    +----+---+

    then we compute the DCG and IDCG scores, using the discount:
    DCG@2 = 3 / log2(2 + 1) + 63 / log2(1 + 1) = 64.89
    IDCG@2 = 15 / log2(1 + 1) + 3 / log2(2 + 1) + 63 / log2(1 + 1) + 7 / log2(2 + 1) = 84.30
    nDCG@2 = 64.89 / 84.30 = 0.77

    For further details, please refer to this `link <https://en.wikipedia.org/wiki/Discounted_cumulative_gain>`_.

    Attributes:
        ndcg (Tensor): The total value of ndcg per user.
        users (Tensor): The number of users evaluated.

    Args:
        k (int): The cutoff.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    ndcg: Tensor
    users: Tensor

    def __init__(
        self, k: int, *args: Any, dist_sync_on_step: bool = False, **kwargs: Any
    ):
        super().__init__(k, dist_sync_on_step)
        self.add_state("ndcg", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor):
        """Updates the metric state with the new batch of predictions."""
        # The discounted relevance is computed as 2^(rel + 1) - 1
        target = self.discounted_relevance(target)
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
