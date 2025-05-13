# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.base_metric import TopKMetric, BaseMetric
from warprec.utils.registry import metric_registry


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
        metric_1 (BaseMetric): First metric to use inside F1-score computation.
        metric_2 (BaseMetric): Second metric to use inside F1-score computation.

    Args:
        k (int): The number of top recommendations to consider (cutoff).
        train_set (csr_matrix): Sparse matrix of training interactions (users x items).
        *args (Any): Additional arguments to pass to the parent class.
        beta (float): The weight of recall in the harmonic mean. Default is 1.0.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    metric_1: BaseMetric
    metric_2: BaseMetric

    def __init__(
        self,
        k: int,
        train_set: csr_matrix,
        *args: Any,
        beta: float = 1.0,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.beta = beta
        self.metric_1 = metric_registry.get(
            "Precision",
            k=k,
            train_set=train_set,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )
        self.metric_2 = metric_registry.get(
            "Recall",
            k=k,
            train_set=train_set,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )

    def update(self, preds: Tensor, target: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        # Update first metric
        self.metric_1.update(preds, target, **kwargs)

        # Update second metric
        self.metric_2.update(preds, target, **kwargs)

    def compute(self):
        """Computes the F1 score using precision and recall."""
        score_1 = self.metric_1.compute().get("Precision", 0)
        score_2 = self.metric_2.compute().get("Recall", 0)

        f1_score = (
            (1 + self.beta**2)
            * (score_1 * score_2)
            / (self.beta**2 * score_1 + score_2)
            if score_1 + score_2 > 0
            else torch.tensor(0.0)
        )
        return {self.name: f1_score}

    def reset(self):
        """Resets the metric state."""
        self.metric_1.reset()
        self.metric_2.reset()
