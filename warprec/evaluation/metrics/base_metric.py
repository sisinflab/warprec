# pylint: disable=arguments-differ
from abc import abstractmethod, ABC
from typing import Any, Tuple, Set

import torch
from torch import Tensor
from torchmetrics import Metric
from scipy.sparse import csr_matrix
from warprec.utils.enums import MetricBlock


class BaseMetric(Metric, ABC):
    """The base definition of a metric using Torchmetrics."""

    _REQUIRED_COMPONENTS: Set[MetricBlock] = (
        set()
    )  # This defines the data that needs to be pre-computed

    @abstractmethod
    def compute(self) -> dict[str, float]:
        pass

    @classmethod
    def binary_relevance(self, target: Tensor) -> Tensor:
        """Compute the binary relevance tensor.

        Args:
            target (Tensor): The target tensor.

        Returns:
            Tensor: The binary relevance tensor.
        """
        return (target > 0).float()

    @classmethod
    def discounted_relevance(self, target: Tensor) -> Tensor:
        """Compute the discounted relevance tensor.

        Args:
            target (Tensor): The target tensor.

        Returns:
            Tensor: The discounted relevance tensor.
        """
        return torch.where(target > 0, 2 ** (target + 1) - 1, target)

    @classmethod
    def valid_users(self, target: Tensor) -> int:
        """Compute the number of valid users.

        Args:
            target (Tensor): The target tensor.

        Returns:
            int: The number of valid users.
        """
        return int((target > 0).any(dim=1).sum().item())

    @classmethod
    def top_k_values_indices(self, preds: Tensor, k: int) -> Tuple[Tensor, Tensor]:
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
        self, target: Tensor, top_k_indices: Tensor
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
    def top_k_relevance(self, preds: Tensor, target: Tensor, k: int) -> Tensor:
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
        self, train_set: csr_matrix, pop_ratio: float = 0.8
    ) -> Tuple[Tensor, Tensor]:
        """Compute popularity as tensors of the short head and long tail.

        Args:
            train_set (csr_matrix): The training interaction data.
            pop_ratio (float): The percentile considered popular.

        Returns:
            Tuple[Tensor, Tensor]:
                - Tensor: The tensor containing indices of short head items.
                - Tensor: The tensor containing indices of long tail items.
        """
        # Compute item frequencies
        item_interactions = torch.tensor(
            train_set.getnnz(axis=0)
        ).float()  # Get number of non-zero elements in each column

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

    def compute_popularity(self, train_set: csr_matrix) -> Tensor:
        """Compute popularity tensor based on the interactions.

        Args:
            train_set (csr_matrix): The training interaction data.

        Returns:
            Tensor: The interaction count for each item.
        """
        # Compute item frequencies
        item_interactions = torch.tensor(
            train_set.getnnz(axis=0)
        ).float()  # Get number of non-zero elements in each column

        # Avoid division by zero: set minimum interaction
        # count to 1 if any item has zero interactions
        item_interactions = torch.clamp(item_interactions, min=1)
        return item_interactions

    def compute_novelty_profile(
        self, train_set: csr_matrix, log_discount: bool = False
    ) -> Tensor:
        """Compute the novelty profile based on the count of interactions.

        Args:
            train_set (csr_matrix): The training interaction data.
            log_discount (bool): Whether or not to compute the discounted novelty.

        Returns:
            Tensor: A tensor that contains the novelty score for each item.
        """
        # Compute item frequencies
        item_interactions = torch.tensor(
            train_set.getnnz(axis=0)
        ).float()  # Get number of non-zero elements in each column
        total_interactions = item_interactions.sum()
        users = train_set.shape[0]

        # Avoid division by zero: set minimum interaction
        # count to 1 if any item has zero interactions
        item_interactions = torch.clamp(item_interactions, min=1)

        # Compute novelty scores
        if log_discount:
            return -torch.log2(item_interactions / total_interactions)
        return 1 - (item_interactions / users)

    @property
    def name(self):
        """The name of the metric."""
        return self.__class__.__name__


class TopKMetric(BaseMetric):
    """The definition of a Top-K metric."""

    def __init__(self, k: int, *args: Any, dist_sync_on_step=False, **kwargs: Any):
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
