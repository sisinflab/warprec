# pylint: disable=arguments-differ, unused-argument, line-too-long
from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import TopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("SRecall")
class SRecall(TopKMetric):
    r"""Subtopic Recall (SRecall) metric for evaluating recommender systems.

    It measures the proportion of a user's relevant features (or subtopics) that are present
    among the top-k recommended items. A higher value indicates that the recommendations
    cover a wider variety of the user's interests (features/subtopics).

    The metric formula for a single user is defined as:

    $$
        \mathrm {SRecall}=\frac{\left|\cup_{i=1}^{K} {subtopics}\left(d_{i}\right)\right|}{n_{A}}
    $$

    where:
        - $K$: The cutoff, i.e., the number of items in the top-k.
        - $d_i$: The i-th item recommended in the top-k.
        - ${subtopics}\left(d_{i}\right)$: The set of features (subtopics) associated with item $d_i$.
        - $\left|\cup_{i=1}^{K} {subtopics}\left(d_{i}\right)\right|$: The cardinality of the union set of features of *relevant* items present in the top-k recommendations for the user. This represents the number of unique relevant features retrieved in the top-k.
        - $n_{A}$: The total number of unique features associated with *all* relevant items for the user.

    The final SRecall metric is calculated as the average of these ratios across all users in the dataset.

    Tensor Calculation Example:

    Consider a batch with 2 users, k=2, and 4 items with 3 features each.
    Item features are defined by the `side_information`:
    Item 0: [1, 0, 1] (Features 0 and 2)
    Item 1: [0, 1, 0] (Feature 1)
    Item 2: [1, 1, 0] (Features 0 and 1)
    Item 3: [0, 0, 1] (Feature 2)

    `preds` (recommendation scores):
    +---+---+---+---+
    | 8 | 2 | 7 | 2 |  (User 1)
    | 5 | 4 | 3 | 9 |  (User 2)
    +---+---+---+---+

    `target` (relevant items > 0):
    +---+---+---+---+
    | 1 | 0 | 1 | 0 |  (User 1: Relevant Items 0, 2)
    | 0 | 0 | 1 | 1 |  (User 2: Relevant Items 2, 3)
    +---+---+---+---+

    Extract the indices of the top-k items (k=2):
    `top_k_indices`
    +---+---+
    | 0 | 2 |  (User 1: Items 0, 2)
    | 3 | 0 |  (User 2: Items 3, 0)
    +---+---+

    Mask of relevant items in top-k (`relevant_top_k_mask`):
    Consider only items that are both in top-k AND relevant.
    User 1: Top-k [Item 0, Item 2]. Relevant [Item 0, Item 2]. Relevant in Top-k [Item 0, Item 2].
    User 2: Top-k [Item 3, Item 0]. Relevant [Item 2, Item 3]. Relevant in Top-k [Item 3].
    +-------+-------+-------+-------+
    | True  | False | True  | False |  (User 1)
    | False | False | False | True  |  (User 2)
    +-------+-------+-------+-------+

    Calculate the number of unique features retrieved in top-k *that are relevant* for each user (Numerator):
    User 1: Relevant items in top-k are Item 0 ([1,0,1]) and Item 2 ([1,1,0]). Unique features among these: {0, 1, 2}. Count: 3.
    User 2: Relevant item in top-k is Item 3 ([0,0,1]). Unique features among these: {2}. Count: 1.
    Numerator for User 1: 3
    Numerator for User 2: 1

    Calculate the total number of unique features associated with *all* relevant items for each user (Denominator $n_A$):
    User 1: Relevant items are Item 0 ([1,0,1]) and Item 2 ([1,1,0]). Unique features among all relevant: {0, 1, 2}. Count ($n_A$): 3.
    User 2: Relevant items are Item 2 ([1,1,0]) and Item 3 ([0,0,1]). Unique features among all relevant: {0, 1, 2}. Count ($n_A$): 3.
    Denominator for User 1: 3
    Denominator for User 2: 3

    Calculate the ratio for each user and sum them:
    User 1 Ratio: 3 / 3 = 1.0
    User 2 Ratio: 1 / 3 = 0.333...
    Sum of ratios (`ratio_feature_retrieved`): 1.0 + 0.333... = 1.333...

    Count the number of users in the batch with at least one relevant item (`users`):
    User 1 has relevant items. User 2 has relevant items. User count: 2.

    Final SRecall = Sum of ratios / Number of users = 1.333... / 2 = 0.666...

    For further details, please refer to the `paper <https://dl.acm.org/doi/abs/10.1145/2795403.2795405>`_.

    Attributes:
        feature_lookup (Tensor): The item feature lookup tensor.
        ratio_feature_retrieved (Tensor): Sum, across all processed users, of the ratio between
            the number of unique relevant features retrieved in the top-k and the total number of unique relevant features.
        users (Tensor): The total number of processed users who have at least one relevant item.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.

    Args:
        k (int): The cutoff for recommendations.
        num_users (int): Number of users in the training set.
        feature_lookup (Tensor): A tensor containing the features associated with each item.
            Tensor shape is expected to be [num_items, num_features].
        *args (Any): Additional positional arguments list.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        dist_sync_on_step (bool): Torchmetrics parameter for distributed synchronization. Defaults to `False`.
        **kwargs (Any): Additional keyword arguments dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_INDICES,
    }
    _CAN_COMPUTE_PER_USER: bool = True

    feature_lookup: Tensor
    ratio_feature_retrieved: Tensor
    users: Tensor
    compute_per_user: bool

    def __init__(
        self,
        k: int,
        num_users: int,
        feature_lookup: Tensor,
        *args: Any,
        compute_per_user: bool = False,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k, dist_sync_on_step=dist_sync_on_step)
        self.compute_per_user = compute_per_user

        if self.compute_per_user:
            self.add_state(
                "ratio_feature_retrieved",
                default=torch.zeros(num_users),
                dist_reduce_fx="sum",
            )  # Initialize a tensor to store metric value for each user
        else:
            self.add_state(
                "ratio_feature_retrieved",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )  # Initialize a scalar to store global value
        self.add_state("users", torch.tensor(0.0), dist_reduce_fx="sum")

        # Add feature lookup as buffer
        self.register_buffer("feature_lookup", feature_lookup)

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Computes the final value of the metric."""
        target = kwargs.get("ground", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_indices: Tensor = kwargs.get(
            f"top_{self.k}_indices", self.top_k_values_indices(preds, self.k)[1]
        )

        # Create a mask for items in the top-k recommendations
        top_k_mask = torch.zeros_like(preds, dtype=torch.bool, device=preds.device)
        top_k_mask.scatter_(
            dim=1, index=top_k_indices, value=True
        )  # [batch_size x num_items]

        # Filter only for relevant items
        relevant_mask = target > 0
        relevant_top_k_mask = top_k_mask & relevant_mask  # [batch_size x num_items]

        # Handle possible filtering of the lookup features
        item_indices = kwargs.get("item_indices", None)
        if item_indices is not None:
            sampled_features = self.feature_lookup[item_indices]
        else:
            sampled_features = self.feature_lookup

        # Compute numerator as the number of features retrieved
        # from the recommender and relevant
        masked_features = sampled_features.unsqueeze(0) * relevant_top_k_mask.unsqueeze(
            -1
        )  # [batch_size x num_items x num_features]
        user_feature_counts = masked_features.sum(dim=1)  # [batch_size x num_features]
        unique_feature_mask = user_feature_counts > 0  # [batch_size x num_features]
        unique_feature_counts = unique_feature_mask.sum(dim=1)  # [batch_size]

        # Compute denominator as the number of features relevant to the user
        relevant_features = sampled_features.unsqueeze(0) * relevant_mask.unsqueeze(
            -1
        )  # [batch_size x num_items x num_features]
        user_relevant_counts = relevant_features.sum(
            dim=1
        )  # [batch_size x num_features]
        unique_relevant_mask = user_relevant_counts > 0  # [batch_size x num_features]
        unique_relevant_counts = unique_relevant_mask.sum(dim=1)  # [batch_size]

        # Update the state avoiding division by zero
        non_zero_mask = unique_relevant_counts != 0
        if self.compute_per_user:
            self.ratio_feature_retrieved.index_add_(
                0,
                user_indices[non_zero_mask],
                unique_feature_counts[non_zero_mask]
                / unique_relevant_counts[non_zero_mask],
            )
        else:
            self.ratio_feature_retrieved += (
                unique_feature_counts[non_zero_mask]
                / unique_relevant_counts[non_zero_mask]
            ).sum()

        # Count only users with at least one interaction
        self.users += users

    def compute(self):
        if self.compute_per_user:
            srecall = self.ratio_feature_retrieved
        else:
            srecall = (
                self.ratio_feature_retrieved / self.users
                if self.users > 0
                else torch.tensor(0.0)
            ).item()
        return {self.name: srecall}
