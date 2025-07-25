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
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.

    Args:
        k (int): The number of top recommendations to consider (cutoff).
        num_users (int): Number of users in the training set.
        num_items (int): Number of items in the training set.
        train_set (csr_matrix): Sparse matrix of training interactions (users x items).
        *args (Any): Additional arguments to pass to the parent class.
        beta (float): The weight of recall in the harmonic mean. Default is 1.0.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        dist_sync_on_step (bool): Torchmetrics parameter.
        metric_name_1 (str): The name of the first metric. Defaults to Precision.
        metric_name_2 (str): The name of the second metric. Defaults to Recall.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    _CAN_COMPUTE_PER_USER: bool = True

    metric_1: BaseMetric
    metric_2: BaseMetric
    compute_per_user: bool

    def __init__(
        self,
        k: int,
        num_users: int,
        num_items: int,
        train_set: csr_matrix,
        *args: Any,
        beta: float = 1.0,
        compute_per_user: bool = False,
        dist_sync_on_step: bool = False,
        metric_name_1: str = "Precision",
        metric_name_2: str = "Recall",
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.num_users = num_users
        self.beta = beta
        self.compute_per_user = compute_per_user
        self.metric_name_1 = metric_name_1
        self.metric_name_2 = metric_name_2

        # Set up metrics
        self.metric_1 = metric_registry.get(
            metric_name_1,
            k=k,
            num_users=num_users,
            num_items=num_items,
            train_set=train_set,
            compute_per_user=True,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )
        self.metric_2 = metric_registry.get(
            metric_name_2,
            k=k,
            num_users=num_users,
            num_items=num_items,
            train_set=train_set,
            compute_per_user=True,
            dist_sync_on_step=dist_sync_on_step,
            **kwargs,
        )

        # Update needed blocks to be the union of the blocks
        # of the two metrics
        self._REQUIRED_COMPONENTS = (
            self.metric_1._REQUIRED_COMPONENTS | self.metric_2._REQUIRED_COMPONENTS
        )

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        # Update first metric
        self.metric_1.update(preds, user_indices, **kwargs)

        # Update second metric
        self.metric_2.update(preds, user_indices, **kwargs)

    def compute(self):
        """Computes the F1 score using the two custom metrics."""
        score_1 = self.metric_1.compute().get(
            self.metric_name_1, torch.zeros(self.num_users)
        )
        score_2 = self.metric_2.compute().get(
            self.metric_name_2, torch.zeros(self.num_users)
        )

        if self.compute_per_user:
            f1_score = (
                (1 + self.beta**2)
                * (score_1 * score_2)
                / (self.beta**2 * score_1 + score_2)
            ).nan_to_num(0)  # In user_wise scenario the denominator might be 0
        else:
            f1_score = (
                (
                    (1 + self.beta**2)
                    * (score_1 * score_2)
                    / (self.beta**2 * score_1 + score_2)
                )
                .nan_to_num(0)
                .mean()
                .item()
            )
        return {self.name: f1_score}

    @property
    def name(self):
        """The name of the metric customized based on the metrics compared."""
        if self.metric_name_1 == "Precision" and self.metric_name_2 == "Recall":
            return self.__class__.__name__
        return f"F1[{self.metric_name_1}, {self.metric_name_2}]"
