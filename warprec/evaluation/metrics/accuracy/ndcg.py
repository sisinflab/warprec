from typing import Any, Set, Tuple

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("nDCG")
class nDCG(UserAverageTopKMetric):
    """The nDCG@k metric is defined as the rapport of the DCG@k and the IDCG@k.

    The DCG@k represent the Discounted Cumulative Gain,
        which measures the gain of the items retrieved.

    The IDCG@k represent the Ideal Discounted Cumulative Gain,
        which measures the maximum gain possible
        obtainable by a perfect model.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.DISCOUNTED_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_DISCOUNTED_RELEVANCE,
    }

    def unpack_inputs(
        self, preds: Tensor, **kwargs: Any
    ) -> Tuple[Tensor, Tensor, Tensor]:
        target = kwargs.get("discounted_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel = kwargs.get(
            f"top_{self.k}_discounted_relevance",
            self.top_k_relevance(preds, target, self.k),
        )
        return target, users, top_k_rel

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        ideal_rel = torch.topk(target, self.k, dim=1).values
        dcg_score = self.dcg(top_k_rel)
        idcg_score = self.dcg(ideal_rel).clamp(min=1e-10)

        return (dcg_score / idcg_score).nan_to_num(0)


@metric_registry.register("nDCGRendle2020")
class nDCGRendle2020(UserAverageTopKMetric):
    """Normalized Discounted Cumulative Gain (nDCG) metric for evaluating recommender systems.

    It measures the ranking quality by considering the position of relevant items,
    giving higher scores to relevant items that appear earlier in the recommendation list.
    This implementation calculates nDCG@k using *binary relevance* (0 or 1).
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        ideal_rel = torch.topk(target, self.k, dim=1).values
        dcg_score = self.dcg(top_k_rel)
        idcg_score = self.dcg(ideal_rel).clamp(min=1e-10)

        return (dcg_score / idcg_score).nan_to_num(0)
