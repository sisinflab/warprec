from typing import Any, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric
from warprec.utils.enums import MetricBlock
from warprec.utils.registry import metric_registry


@metric_registry.register("SRecall")
class SRecall(UserAverageTopKMetric):
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
        - $\left|\cup_{i=1}^{K} {subtopics}\left(d_{i}\right)\right|$: The cardinality of the union set of
            features of *relevant* items present in the top-k recommendations for the user. This represents
            the number of unique relevant features retrieved in the top-k.
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

    Args:
        k (int): The cutoff for recommendations.
        num_users (int): Number of users in the training set.
        feature_lookup (Tensor): A tensor containing the features associated with each item.
            Tensor shape is expected to be [num_items, num_features].
        dist_sync_on_step (bool): Torchmetrics parameter for distributed synchronization. Defaults to `False`.
        **kwargs (Any): Additional keyword arguments dictionary.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.BINARY_RELEVANCE,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_INDICES,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    feature_lookup: Tensor

    def __init__(
        self,
        k: int,
        num_users: int,
        feature_lookup: Tensor,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k=k, num_users=num_users, dist_sync_on_step=dist_sync_on_step)

        # Add feature lookup as buffer
        self.register_buffer("feature_lookup", feature_lookup)

    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        top_k_indices = kwargs.get(f"top_{self.k}_indices")
        item_indices = kwargs.get("item_indices")

        # Handle sampled item indices if provided
        if item_indices is not None:
            # We subset the feature lookup to match the batch items
            batch_features = self.feature_lookup[
                item_indices
            ]  # [batch, num_samples, n_feats]

            # For Top-K, we need to map indices to gather features
            top_k_features = torch.gather(
                batch_features,
                1,
                top_k_indices.unsqueeze(-1).expand(-1, -1, batch_features.size(-1)),
            )  # [batch, k, n_feats]

        else:
            batch_features = self.feature_lookup.unsqueeze(0)  # [1, num_items, n_feats]
            top_k_features = self.feature_lookup[top_k_indices]  # [batch, k, n_feats]

        # Denominator: Unique features in ALL Relevant items
        relevant_mask = (target > 0).unsqueeze(-1)  # [batch, num_items, 1]

        # Mask features that are not relevant
        relevant_features_batch = batch_features * relevant_mask

        # Count unique features: Sum over items -> if > 0, feature is present
        # [batch, n_feats]
        features_present_in_relevant = (relevant_features_batch.sum(dim=1) > 0).float()

        # Total unique relevant features per user
        denominator = features_present_in_relevant.sum(dim=1)  # [batch]

        # Numerator: Unique features in Top-K AND Relevant items
        top_k_rel_mask = (top_k_rel > 0).unsqueeze(-1)  # [batch, k, 1]

        # Mask features of Top-K items that are NOT relevant
        relevant_top_k_features = top_k_features * top_k_rel_mask

        # Count unique features
        features_present_in_top_k = (relevant_top_k_features.sum(dim=1) > 0).float()

        # Total unique relevant features retrieved per user
        numerator = features_present_in_top_k.sum(dim=1)  # [batch]

        # Compute Ratio
        # Handle division by zero (users with no relevant items)
        return torch.where(
            denominator > 0,
            numerator / denominator,
            torch.tensor(0.0, device=preds.device),
        )
