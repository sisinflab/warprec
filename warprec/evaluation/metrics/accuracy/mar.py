# pylint: disable=arguments-differ, unused-argument, line-too-long, duplicate-code
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("MAR")
class MAR(TopKMetric):
    """Mean Average Recall (MAR) at K.

    MAR@K calculates the mean of the Average Recall for all users.

    The metric formula is defined as:
        MAR@K = sum_{u=1}^{n_users} sum_{i=1}^{k} (R@i * rel_{u,i}) / n_users

    where:
        - R@i is the recall at i-th position.
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

    The recall at i-th position is calculated as the sum of the relevant items
    divided by the number of relevant items:
       RECALL
    +----+----+
    | 0  | .5 |
    | .5 | 0  |
    +----+----+

    the normalization is the minimum between the number of relevant items and k:
    NORMALIZATION
    +---+---+
    | 2 | 2 |
    +---+---+

    MAR@2 = 0.5 / 2 + 0.5 / 2 = 0.5

    For further details, please refer to this `link <https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#So-Why-Did-I-Bother-Defining-Recall?>`_.

    Attributes:
        ar (Tensor): The average recall tensor.
        users (Tensor): The number of users evaluated.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.

    Args:
        k (int): The recommendation cutoff.
        num_users (int): Number of users in the training set.
        *args (Any): Additional arguments to pass to the parent class.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }
    _CAN_COMPUTE_PER_USER: bool = True

    ar: Tensor
    users: Tensor
    compute_per_user: bool

    def __init__(
        self,
        k: int,
        num_users: int,
        *args: Any,
        compute_per_user: bool = False,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step)
        self.compute_per_user = compute_per_user

        if self.compute_per_user:
            self.add_state(
                "ar", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )  # Initialize a tensor to store metric value for each user
        else:
            self.add_state(
                "ar", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )  # Initialize a scalar to store global value
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Updates the MAR metric state with a batch of predictions."""
        target: Tensor = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel: Tensor = kwargs.get(
            f"top_{self.k}_binary_relevance",
            self.top_k_relevance(preds, target, self.k),
        )

        recall_at_i = top_k_rel.cumsum(dim=1) / target.sum(dim=1).unsqueeze(1).clamp(
            min=1
        )  # [batch_size, k]
        normalization = torch.minimum(
            target.sum(dim=1),
            torch.tensor(self.k, dtype=target.dtype, device=target.device),
        )  # [batch_size]

        # Compute AR per user
        ar_per_user = torch.where(
            normalization > 0,
            (recall_at_i * top_k_rel).sum(dim=1) / normalization,
            torch.tensor(0.0, device=self._device),
        )  # [batch_size]

        if self.compute_per_user:
            self.ar.index_add_(
                0, user_indices, ar_per_user
            )  # Index metric values per user
        else:
            self.ar += ar_per_user.sum()  # Compute total average recall

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final MAR@K value."""
        if self.compute_per_user:
            mar = self.ar  # Return the tensor with per_user metric
        else:
            mar = (
                self.ar / self.users if self.users > 0 else torch.tensor(0.0)
            ).item()  # Return the metric value
        return {self.name: mar}
