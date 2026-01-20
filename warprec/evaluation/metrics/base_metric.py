from abc import abstractmethod, ABC
from typing import Any, Tuple, Set, Optional

import torch
from torch import Tensor
from torchmetrics import Metric
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

        # Add padding row
        padding_row = torch.zeros(1, device=item_interactions.device)
        item_interactions = torch.cat((item_interactions, padding_row), dim=0)

        # Compute novelty scores
        if log_discount:
            return -torch.log2(item_interactions / total_interactions).unsqueeze(0)
        return (1 - (item_interactions / num_users)).unsqueeze(0)

    def compute_area_stats(
        self, preds: Tensor, target: Tensor, num_items: int, k: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """Computes the Area per user and the Number of Positives per user.

        Args:
            preds (Tensor): Predictions tensor.
            target (Tensor): Binary relevance tensor.
            num_items (int): Total number of items.
            k (Optional[int]): Cutoff for top-k evaluation. If None, considers all items.

        Returns:
            Tuple[Tensor, Tensor]:
                - Tensor: Area per user.
                - Tensor: Number of positives per user.
        """
        device = preds.device
        batch_size = preds.shape[0]

        # Negative samples count
        # Logic: Total - Train(masked) - Target + 1
        train_set = torch.isinf(preds).logical_and(preds < 0).sum(dim=1)
        target_set = target.sum(dim=1)
        neg_num = (num_items - train_set - target_set + 1).unsqueeze(1)

        # Sorting
        _, sorted_preds = torch.sort(preds, dim=1, descending=True)
        sorted_target = torch.gather(target, 1, sorted_preds)

        # Optional Slicing for top-l
        if k is not None:
            sorted_target = sorted_target[:, :k]

        # Effective Rank and Progressive Position
        # Create column indices [0, 1, 2, ...]
        col_indices = torch.arange(sorted_target.shape[1], device=device).expand(
            batch_size, -1
        )

        effective_rank = torch.where(
            sorted_target == 1, col_indices, torch.tensor(0.0, device=device)
        )

        cumsum = torch.cumsum(sorted_target, dim=1)
        progressive_position = torch.where(
            sorted_target == 1, cumsum - 1, sorted_target
        )

        # AUC Matrix Calculation
        # Formula: (Neg - Eff + Prog) / Neg
        auc_matrix = torch.where(
            sorted_target > 0,
            ((neg_num - effective_rank + progressive_position) / neg_num),
            sorted_target,  # This puts 0 where target is 0
        )

        # Aggregation per user
        area_per_user = auc_matrix.sum(dim=1)
        positives_per_user = sorted_target.sum(dim=1)

        return area_per_user, positives_per_user

    @property
    def name(self):
        """The name of the metric."""
        return self.__class__.__name__

    @property
    def components(self):
        """The required components to compute the metric."""
        return self._REQUIRED_COMPONENTS


class RatingMetric(BaseMetric):
    """The definition of Rating Metric.

    Attributes:
        error_sum (Tensor): The tensor to store per-user error sum.
        total_count (Tensor): The tensor to store per-user count of ratings.

    Args:
        num_users (int): Number of users in the training set.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    error_sum: Tensor
    total_count: Tensor

    def __init__(
        self,
        num_users: int,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "error_sum", default=torch.zeros(num_users), dist_reduce_fx="sum"
        )  # Initialize a tensor to store per-user error sum
        self.add_state(
            "total_count", default=torch.zeros(num_users), dist_reduce_fx="sum"
        )  # Initialize a tensor to store per-user count of ratings

    @abstractmethod
    def _compute_element_error(self, preds: Tensor, target: Tensor) -> Tensor:
        """Computes the error between predictions and target."""
        pass

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Unified update logic using index_add_."""
        target = kwargs.get("ground", torch.zeros_like(preds))

        # Mask for valid ratings
        mask = target > 0

        # Compute error
        errors = self._compute_element_error(preds, target)

        # Zero out errors for non-rated items to be safe
        errors = errors * mask.float()

        # Accumulate per user
        self.error_sum.index_add_(0, user_indices, errors.sum(dim=1))
        self.total_count.index_add_(0, user_indices, mask.sum(dim=1).float())

    def compute(self):
        """Computes the final metric value."""
        results = self.error_sum / self.total_count  # Calculate metric per user
        results[self.total_count == 0] = float(
            "nan"
        )  # Set nan for users with no interactions
        return {self.name: results}


class TopKMetric(BaseMetric):
    """The definition of a Top-K metric.

    Attributes:
        k (int): The cutoff value.

    Args:
        k (int): The cutoff for recommendations.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    k: int

    def __init__(
        self,
        k: int,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
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


class UserAverageTopKMetric(TopKMetric):
    """The definition of a User Average Top-K metric.

    Attributes:
        scores (Tensor): The tensor to store metric values.
        user_interactions (Tensor): The tensor to store number of interactions per user.

    Args:
        k (int): The cutoff.
        num_users (int): Number of users in the training set.
        *args (Any): The argument list.
        dist_sync_on_step (bool): Torchmetrics parameter.
        **kwargs (Any): Additional keyword arguments to pass to the parent class.
    """

    scores: Tensor
    user_interactions: Tensor

    def __init__(
        self,
        k: int,
        num_users: int,
        *args: Any,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(k=k, dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "scores", default=torch.zeros(num_users), dist_reduce_fx="sum"
        )  # Initialize a tensor to store metric value for each user
        self.add_state(
            "user_interactions", default=torch.zeros(num_users), dist_reduce_fx="sum"
        )  # Initialize a tensor to store number of interactions per user

    def unpack_inputs(
        self, preds: Tensor, **kwargs: Any
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Default unpacking method used by most metrics.

        Retrieves the binary relevance, valid users and top-k binary relevance.

        Args:
            preds (Tensor): The prediction tensor.
            **kwargs (Any): The keyword argument dictionary.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - Tensor: The target tensor.
                - Tensor: The valid users tensor.
                - Tensor: The top-k relevance tensor.
        """
        target = kwargs.get("binary_relevance", torch.zeros_like(preds))
        users = kwargs.get("valid_users", self.valid_users(target))
        top_k_rel = kwargs.get(
            f"top_{self.k}_binary_relevance",
            self.top_k_relevance(preds, target, self.k),
        )
        return target, users, top_k_rel

    @abstractmethod
    def compute_scores(
        self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
    ) -> Tensor:
        """Math formula for the specific metric.

        Metrics must implement this method.

        Args:
            preds (Tensor): The prediction tensor.
            target (Tensor): The target tensor.
            top_k_rel (Tensor): The top-k relevance tensor.
            **kwargs (Any): The keyword argument dictionary.

        Returns:
            Tensor: The computed metric values per user.
        """
        pass

    def update(self, preds: Tensor, user_indices: Tensor, **kwargs: Any):
        """Unified update logic."""
        target, users, top_k_data = self.unpack_inputs(preds, **kwargs)
        batch_scores = self.compute_scores(preds, target, top_k_data, **kwargs)

        # Safety masking
        batch_scores = torch.where(
            users > 0, batch_scores, torch.tensor(0.0, device=preds.device)
        )

        # Accumulate per user
        self.scores.index_add_(0, user_indices, batch_scores)
        self.user_interactions.index_add_(0, user_indices, users)

    def compute(self):
        """Computes the final metric value."""
        scores = self.scores / self.user_interactions  # Normalize the metric score
        scores[self.user_interactions == 0] = float(
            "nan"
        )  # Set nan for users with no interactions
        return {self.name: scores}
