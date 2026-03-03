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
