# pylint: disable=arguments-differ
from abc import abstractmethod, ABC
from typing import Any, Tuple, Set

import torch
from torch import Tensor
from torchmetrics import Metric
from warprec.utils.enums import MetricBlock


class BaseMetric(Metric, ABC):
    """The base definition of a metric using Torchmetrics."""

    _REQUIRED_COMPONENTS: Set[MetricBlock] = (
        set()
    )  # This defines the data that needs to be pre-computed
    _CAN_COMPUTE_PER_USER: bool = (
        False  # Flag value for user-wise computation of metric
    )

    @abstractmethod
    def compute(self) -> dict[str, float]:
        pass

    @classmethod
    def binary_relevance(cls, target: Tensor) -> Tensor:
        """Compute the binary relevance tensor.

        Args:
            target (Tensor): The target tensor.

        Returns:
            Tensor: The binary relevance tensor.
        """
        return (target > 0).float()

    @classmethod
    def discounted_relevance(cls, target: Tensor) -> Tensor:
        """Compute the discounted relevance tensor.

        Args:
            target (Tensor): The target tensor.

        Returns:
            Tensor: The discounted relevance tensor.
        """
        return torch.where(target > 0, 2 ** (target + 1) - 1, target)

    @classmethod
    def valid_users(cls, target: Tensor) -> Tensor:
        """Compute the number of valid users.

        Args:
            target (Tensor): The target tensor.

        Returns:
            Tensor: A Tensor containing 1 if a user is valid
                or 0 otherwise.
        """
        return (target > 0).any(dim=1).float()

    @classmethod
    def top_k_values_indices(cls, preds: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """Compute the top k indices and values.

        Args:
            preds (Tensor): The prediction tensor
            k (int): The value of cutoff.

        Returns:
            Tuple[Tensor, Tensor]:
                - Tensor: The values tensor.
                - Tensor: The indices tensor
        """
        return torch.topk(preds, k, dim=1)

    @classmethod
    def top_k_relevance_from_indices(
        cls, target: Tensor, top_k_indices: Tensor
    ) -> Tensor:
        """Compute the top k relevance tensor.

        Args:
            target (Tensor): The target tensor.
            top_k_indices (Tensor): The top k indices.

        Returns:
            Tensor: The top k relevance tensor.
        """
        return torch.gather(target, dim=1, index=top_k_indices)

    @classmethod
    def top_k_relevance(cls, preds: Tensor, target: Tensor, k: int) -> Tensor:
        """Compute the top k relevance tensor.

        Args:
            preds (Tensor): The prediction tensor
            target (Tensor): The target tensor.
            k (int): The value of cutoff.

        Returns:
            Tensor: The top k relevance tensor.
        """
        _, top_k_indices = torch.topk(preds, k, dim=1)
        return torch.gather(target, dim=1, index=top_k_indices)

    def compute_head_tail(
        self, item_interactions: Tensor, pop_ratio: float = 0.8
    ) -> Tuple[Tensor, Tensor]:
        """Compute popularity as tensors of the short head and long tail.

        Args:
            item_interactions (Tensor): The counts for item interactions in training set.
            pop_ratio (float): The percentile considered popular.

        Returns:
            Tuple[Tensor, Tensor]:
                - Tensor: The tensor containing indices of short head items.
                - Tensor: The tensor containing indices of long tail items.
        """
        # Order item popularity
        sorted_interactions, sorted_indices = torch.sort(
            item_interactions, descending=True
        )

        # Determine short head cutoff based on cumulative popularity
        cumulative_pop = torch.cumsum(sorted_interactions, dim=0)
        total_interactions = item_interactions.sum()
        cutoff_index = torch.where(cumulative_pop > total_interactions * pop_ratio)[0][
            0
        ]

        # Extract indexes from sorted interactions
        short_head_indices = sorted_indices[
            : cutoff_index + 1
        ]  # Include the item at the cutoff
        long_tail_indices = sorted_indices[cutoff_index + 1 :]

        return short_head_indices, long_tail_indices

    def compute_popularity(self, item_interactions: Tensor) -> Tensor:
        """Compute popularity tensor based on the interactions.

        Args:
            item_interactions (Tensor): The counts for item interactions in training set.

        Returns:
            Tensor: The interaction count for each item.
        """
        # Avoid division by zero: set minimum interaction
        # count to 1 if any item has zero interactions
        item_interactions = torch.clamp(item_interactions, min=1)
        return item_interactions

    def compute_novelty_profile(
        self, item_interactions: Tensor, num_users: int, log_discount: bool = False
    ) -> Tensor:
        """Compute the novelty profile based on the count of interactions.

        Args:
            item_interactions (Tensor): The counts for item interactions in training set.
            num_users (int): Number of users in the training set.
            log_discount (bool): Whether or not to compute the discounted novelty.

        Returns:
            Tensor: A tensor that contains the novelty score for each item.
        """
        total_interactions = item_interactions.sum()

        # Avoid division by zero: set minimum interaction
        # count to 1 if any item has zero interactions
        item_interactions = torch.clamp(item_interactions, min=1)

        # Compute novelty scores
        if log_discount:
            return -torch.log2(item_interactions / total_interactions).unsqueeze(0)
        return (1 - (item_interactions / num_users)).unsqueeze(0)

    @property
    def name(self):
        """The name of the metric."""
        return self.__class__.__name__

    @property
    def components(self):
        """The required components to compute the metric."""
        return self._REQUIRED_COMPONENTS


class TopKMetric(BaseMetric):
    """The definition of a Top-K metric."""

    def __init__(self, k: int, dist_sync_on_step=False, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k

    def dcg(self, rel: Tensor) -> Tensor:
        """The Discounted Cumulative Gain definition.

        Args:
            rel (Tensor): The relevance tensor.

        Returns:
            Tensor: The discounted tensor.
        """
        return (
            rel / torch.log2(torch.arange(2, rel.size(1) + 2, device=rel.device))
        ).sum(dim=1)

    def discounted_sum(self, k: int) -> Tensor:
        """Computes the discounted sum for k values.

        Args:
            k (int): The length of the tensor to discount.

        Returns:
            Tensor: The sum of the discounts for k values.
        """
        ranks = torch.arange(k)
        return torch.sum(1.0 / torch.log2(ranks.float() + 2))
