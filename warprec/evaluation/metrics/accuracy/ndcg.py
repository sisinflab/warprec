# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


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
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.

    Args:
        k (int): The cutoff.
        num_users (int): Number of users in the training set.
        *args (Any): The argument list.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): The keyword argument dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.DISCOUNTED_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_DISCOUNTED_RELEVANCE,
    }
    _CAN_COMPUTE_PER_USER: bool = True

    ndcg: Tensor
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
                "ndcg", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )  # Initialize a tensor to store metric value for each user
        else:
            self.add_state(
                "ndcg", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )  # Initialize a scalar to store global value
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        # The discounted relevance is computed as 2^(rel + 1) - 1
        target: Tensor = kwargs.get("discounted_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel: Tensor = kwargs.get(
            f"top_{self.k}_discounted_relevance",
            self.top_k_relevance(preds, target, self.k),
        )

        ideal_rel = torch.topk(target, self.k, dim=1).values
        dcg_score = self.dcg(top_k_rel)
        idcg_score = self.dcg(ideal_rel).clamp(min=1e-10)

        # NOTE: nan_to_num(0) is used in case of division by 0
        if self.compute_per_user:
            self.ndcg.index_add_(
                0, user_indices, (dcg_score / idcg_score).nan_to_num(0)
            )  # Index metric values per user
        else:
            self.ndcg += (
                (dcg_score / idcg_score).nan_to_num(0).sum()
            )  # Sum global nDCG value

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final metric value."""
        if self.compute_per_user:
            ndcg = self.ndcg  # Return the tensor with per_user metric
        else:
            ndcg = (
                self.ndcg / self.users if self.users > 0 else torch.tensor(0.0)
            ).item()  # Return the metric value
        return {self.name: ndcg}


@metric_registry.register("nDCGRendle2020")
class nDCGRendle2020(TopKMetric):
    r"""Normalized Discounted Cumulative Gain (nDCG) metric for evaluating recommender systems.

    It measures the ranking quality by considering the position of relevant items,
    giving higher scores to relevant items that appear earlier in the recommendation list.
    This implementation calculates nDCG@k using *binary relevance* (0 or 1).

    The nDCG@k formula is defined as:

    $$
        nDCG_k = \frac{DCG_k}{IDCG_k}
    $$

    where $DCG_k$ is the Discounted Cumulative Gain at cutoff $k$, calculated as:

    $$
        DCG_k = \sum_{i=1}^k \frac{rel_i}{\log_2(i+1)}
    $$

    and $IDCG_k$ is the Ideal Discounted Cumulative Gain at cutoff $k$.

    In these formulas:
        - $k$: The cutoff, i.e., the number of items considered in the top of the ranked list.
        - $rel_i$: The *binary* relevance (1 if relevant, 0 otherwise) of the item at rank $i$ in the recommendation list.
        - $IDCG_k$: The maximum possible DCG score for the given list of relevant items, achieved by ranking all relevant items before non-relevant items up to rank $k$.

    The metric computes the nDCG@k for each user and returns the average across all users with at least one relevant item.

    Tensor Calculation Example:

    Consider a batch with 2 users, k=3, and 5 items.
    `preds` (recommendation scores):
    +---+---+---+---+---+
    | 5 | 8 | 3 | 6 | 2 |  (User 1)
    | 1 | 9 | 4 | 7 | 3 |  (User 2)
    +---+---+---+---+---+

    `target` (binary relevance, 1 if relevant, 0 otherwise):
    +---+---+---+---+---+
    | 0 | 1 | 0 | 1 | 0 |  (User 1: Relevant Item 1, Item 3)
    | 1 | 0 | 0 | 1 | 0 |  (User 2: Relevant Item 0, Item 3)
    +---+---+---+---+---+

    Extract the indices of the top-k items (k=3):
    `top_k_indices`
    +---+---+---+
    | 1 | 3 | 0 |  (User 1: Items 1, 3, 0)
    | 1 | 3 | 4 |  (User 2: Items 1, 3, 4)
    +---+---+---+

    Get the binary relevance (`rel`) for the items at these top-k indices:
    `rel`
    +---+---+---+
    | 1 | 1 | 0 |  (User 1: Relevance of Items 1, 3, 0 is 1, 1, 0)
    | 0 | 1 | 0 |  (User 2: Relevance of Items 1, 3, 4 is 0, 1, 0)
    +---+---+---+

    Get the ideal binary relevance (`ideal_rel`) by sorting the `target` and taking the top-k:
    User 1 target sorted: [1, 1, 0, 0, 0]. Top-k (k=3): [1, 1, 0].
    User 2 target sorted: [1, 1, 0, 0, 0]. Top-k (k=3): [1, 1, 0].
    `ideal_rel`
    +---+---+---+
    | 1 | 1 | 0 |  (User 1)
    | 1 | 1 | 0 |  (User 2)
    +---+---+---+

    Calculate DCG for `rel` and `ideal_rel` for each user using $\sum_{i=1}^k \frac{rel_i}{\log_2(i+1)}$:
    User 1 DCG (`rel` [1, 1, 0]): $\frac{1}{\log_2(1+1)} + \frac{1}{\log_2(2+1)} + \frac{0}{\log_2(3+1)} = \frac{1}{1} + \frac{1}{\log_2 3} + 0 \approx 1 + 0.6309 = 1.6309$
    User 1 IDCG (`ideal_rel` [1, 1, 0]): $\frac{1}{1} + \frac{1}{\log_2 3} + \frac{0}{\log_2 4} \approx 1.6309$
    User 2 DCG (`rel` [0, 1, 0]): $\frac{0}{\log_2(1+1)} + \frac{1}{\log_2(2+1)} + \frac{0}{\log_2(3+1)} = 0 + \frac{1}{\log_2 3} + 0 \approx 0.6309$
    User 2 IDCG (`ideal_rel` [1, 1, 0]): $\frac{1}{1} + \frac{1}{\log_2 3} + \frac{0}{\log_2 4} \approx 1.6309$

    Calculate nDCG for each user (DCG / IDCG):
    User 1 nDCG: $1.6309 / 1.6309 \approx 1.0$
    User 2 nDCG: $0.6309 / 1.6309 \approx 0.3869$

    Sum nDCG scores (`self.ndcg`): $1.0 + 0.3869 = 1.3869$
    Count users with relevant items (`self.users`): User 1 has relevant items, User 2 has relevant items. Count: 2.

    Final nDCG = Sum of nDCG / Number of users = $1.3869 / 2 \approx 0.69345$

    This implementation provides a standard calculation of nDCG@k often used in the evaluation
        of recommender systems, including contexts related to the work of S. Rendle (e.g., in implicit feedback scenarios with binary relevance).

    For further details, please refer to this `link <https://dl.acm.org/doi/10.1145/3394486.3403226>`_.

    Attributes:
        ndcg (Tensor): The accumulated sum of per-user nDCG scores across all processed batches.
        users (Tensor): The total number of users processed who have at least one relevant item (i.e., contribute to the denominator in the final average).
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.

    Args:
        k (int): The cutoff for recommendations.
        num_users (int): Number of users in the training set.
        *args (Any): Additional positional arguments list.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        dist_sync_on_step (bool): Torchmetrics parameter for distributed synchronization. Defaults to `False`.
        **kwargs (Any): Additional keyword arguments dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }
    _CAN_COMPUTE_PER_USER: bool = True

    ndcg: Tensor
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
                "ndcg", default=torch.zeros(num_users), dist_reduce_fx="sum"
            )  # Initialize a tensor to store metric value for each user
        else:
            self.add_state(
                "ndcg", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )  # Initialize a scalar to store global value
        self.add_state("users", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Updates the metric state with the new batch of predictions."""
        # The discounted relevance is computed as 2^(rel + 1) - 1
        target = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel: Tensor = kwargs.get(
            f"top_{self.k}_binary_relevance",
            self.top_k_relevance(preds, target, self.k),
        )

        ideal_rel = torch.topk(target, self.k, dim=1, largest=True, sorted=True).values

        dcg_score = self.dcg(top_k_rel)
        idcg_score = self.dcg(ideal_rel).clamp(min=1e-10)

        # NOTE: nan_to_num(0) is used in case of division by 0
        if self.compute_per_user:
            self.ndcg.index_add_(
                0, user_indices, (dcg_score / idcg_score).nan_to_num(0)
            )  # Index metric values per user
        else:
            self.ndcg += (
                (dcg_score / idcg_score).nan_to_num(0).sum()
            )  # Sum global nDCG value

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        """Computes the final metric value."""
        if self.compute_per_user:
            ndcg = self.ndcg  # Return the tensor with per_user metric
        else:
            ndcg = (
                self.ndcg / self.users if self.users > 0 else torch.tensor(0.0)
            ).item()  # Return the metric value
        return {self.name: ndcg}
