# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.evaluation.metrics.base_metric import TopKMetric, BaseMetric
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

    This metric also supports the computation of F1-score using different metrics other than
    Precision@k and Recall@k.

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
        metric_name_1 (str): The name of the first metric. Defaults to Precision.
        metric_name_2 (str): The name of the second metric. Defaults to Recall.
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
        metric_name_1: str = "Precision",
        metric_name_2: str = "Recall",
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.beta = beta
        self.metric_name_1 = metric_name_1
        self.metric_name_2 = metric_name_2

        # Set up metrics
        self.metric_1 = metric_registry.get(
            metric_name_1,
            k=k,
            train_set=train_set,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )
        self.metric_2 = metric_registry.get(
            metric_name_2,
            k=k,
            train_set=train_set,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )

        # Update needed blocks to be the union of the blocks
        # of the two metrics
        self._REQUIRED_COMPONENTS = (
            self.metric_1._REQUIRED_COMPONENTS | self.metric_2._REQUIRED_COMPONENTS
        )

    def update(self, preds: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        # Update first metric
        self.metric_1.update(preds, **kwargs)

        # Update second metric
        self.metric_2.update(preds, **kwargs)

    def compute(self):
        """Computes the F1 score using precision and recall."""
        score_1 = self.metric_1.compute().get(self.metric_name_1, 0)
        score_2 = self.metric_2.compute().get(self.metric_name_2, 0)

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

    @property
    def name(self):
        """The name of the metric."""
        if self.metric_name_1 == "Precision" and self.metric_name_2 == "Recall":
            return self.__class__.__name__
        return f"F1[{self.metric_name_1}, {self.metric_name_2}]"
