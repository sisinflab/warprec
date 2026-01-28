# pylint: disable = too-few-public-methods
from typing import List, Any
from abc import ABC

import time
import narwhals as nw
from narwhals.typing import FrameT

from warprec.utils.logger import logger
from warprec.utils.registry import filter_registry


# pylint: disable = unused-argument
class Filter(ABC):
    """Abstract definition of a filter.
    Filters are used to process datasets by applying specific conditions
    or transformations to the data.

    Args:
        user_id_label (str): Column name for user IDs.
        item_id_label (str): Column name for item IDs.
        rating_label (str): Column name for ratings.
        timestamp_label (str): Column name for timestamps.
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(
        self,
        user_id_label: str = "user_id",
        item_id_label: str = "item_id",
        rating_label: str = "rating",
        timestamp_label: str = "timestamp",
        **kwargs: Any,
    ):
        self.user_label = user_id_label
        self.item_label = item_id_label
        self.rating_label = rating_label
        self.timestamp_label = timestamp_label

    def __call__(self, dataset: FrameT) -> FrameT:
        """Apply the filter to the dataset."""
        raise NotImplementedError("Subclasses should implement this method.")


@filter_registry.register("MinRating")
class MinRating(Filter):
    """Filter to select rows based on a minimum rating.

    Args:
        min_rating (float): The minimum rating threshold.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ValueError: If the provided argument is invalid.
    """

    def __init__(self, min_rating: float, **kwargs: Any):
        super().__init__(**kwargs)
        if min_rating <= 0:
            raise ValueError("min_rating must be a positive float.")
        self.min_rating = min_rating

    def __call__(self, dataset: FrameT) -> FrameT:
        """Select rows where the 'rating' column is greater than or equal to min_rating.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only rows with 'rating' >= min_rating.
        """
        return dataset.filter(nw.col(self.rating_label) >= self.min_rating)


@filter_registry.register("UserAverage")
class UserAverage(Filter):
    """Filter to select users based on their average rating.

    Args:
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def __call__(self, dataset: FrameT) -> FrameT:
        """Select rows where the 'rating' column is greater than the user average.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only rows with 'rating' > user average.
        """
        return dataset.filter(
            nw.col(self.rating_label) > nw.col(self.rating_label).mean().over(self.user_label)
        )


@filter_registry.register("ItemAverage")
class ItemAverage(Filter):
    """Filter to select interactions for an item based on the item's average rating.

    Args:
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def __call__(self, dataset: FrameT) -> FrameT:
        """Select rows where the 'rating' column is greater than the item average.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only rows with 'rating' > item average.
        """
        return dataset.filter(
            nw.col(self.rating_label) > nw.col(self.rating_label).mean().over(self.item_label)
        )


@filter_registry.register("UserMin")
class UserMin(Filter):
    """Filter to select users based on a minimum number of interactions.

    Args:
        min_interactions (int): Minimum number of interactions per user.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ValueError: If the provided argument is invalid.
    """

    def __init__(self, min_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if min_interactions <= 0:
            raise ValueError("min_interactions must be a positive integer.")
        self.min_interactions = min_interactions

    def __call__(self, dataset: FrameT) -> FrameT:
        """Select users with at least min_interactions.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only users with interactions >= min_interactions.
        """    
        return dataset.filter(
            nw.len().over(self.user_label) >= self.min_interactions
        )


@filter_registry.register("UserMax")
class UserMax(Filter):
    """Filter to select users based on a maximum number of interactions.

    Args:
        max_interactions (int): Maximum number of interactions per user.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ValueError: If the provided argument is invalid.
    """

    def __init__(self, max_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if max_interactions <= 0:
            raise ValueError("max_interactions must be a positive integer.")
        self.max_interactions = max_interactions

    def __call__(self, dataset: FrameT) -> FrameT:
        """Select users with at most max_interactions.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only users with interactions <= max_interactions.
        """
        return dataset.filter(
            nw.len().over(self.user_label) <= self.max_interactions
        )


@filter_registry.register("ItemMin")
class ItemMin(Filter):
    """Filter to select items based on a minimum number of interactions.

    Args:
        min_interactions (int): Minimum number of interactions per item.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ValueError: If the provided argument is invalid.
    """

    def __init__(self, min_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if min_interactions <= 0:
            raise ValueError("min_interactions must be a positive integer.")
        self.min_interactions = min_interactions

    def __call__(self, dataset: FrameT) -> FrameT:
        """Select items with at least min_interactions.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only items with interactions >= min_interactions.
        """
        return dataset.filter(
            nw.len().over(self.item_label) >= self.min_interactions
        )


@filter_registry.register("ItemMax")
class ItemMax(Filter):
    """Filter to select items based on a maximum number of interactions.

    Args:
        max_interactions (int): Maximum number of interactions per item.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ValueError: If the provided argument is invalid.
    """

    def __init__(self, max_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if max_interactions <= 0:
            raise ValueError("max_interactions must be a positive integer.")
        self.max_interactions = max_interactions

    def __call__(self, dataset: FrameT) -> FrameT:
        """Select items with at most max_interactions.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only items with interactions <= max_interactions.
        """
        return dataset.filter(
            nw.len().over(self.item_label) <= self.max_interactions
        )


@filter_registry.register("IterativeKCore")
class IterativeKCore(Filter):
    """Iteratively apply k-core filtering to the dataset.

    Args:
        min_interactions (int): Minimum number of interactions for users/items.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ValueError: If the provided argument is invalid.
    """

    def __init__(self, min_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if min_interactions <= 0:
            raise ValueError("min_interactions must be a positive integer.")
        self.user_core = UserMin(min_interactions, **kwargs)
        self.item_core = ItemMin(min_interactions, **kwargs)

    def __call__(self, dataset: FrameT) -> FrameT:
        """Apply k-core filtering iteratively until no more users or items can be removed.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset after applying k-core filtering.
        """
        while True:
            # Get current size
            start_len = dataset.select(nw.len()).item()
            
            dataset = self.user_core(dataset)
            dataset = self.item_core(dataset)
            
            end_len = dataset.select(nw.len()).item()

            if end_len == start_len:
                break

        return dataset


@filter_registry.register("NRoundsKCore")
class NRoundsKCore(Filter):
    """Apply k-core filtering for a specified number of rounds.

    Args:
        rounds (int): Number of rounds to apply k-core filtering.
        min_interactions (int): Minimum number of interactions for users/items.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ValueError: If the provided argument is invalid.
    """

    def __init__(self, rounds: int, min_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if rounds <= 0:
            raise ValueError("rounds must be a positive integer.")
        if min_interactions <= 0:
            raise ValueError("min_interactions must be a positive integer.")
        self.user_core = UserMin(min_interactions, **kwargs)
        self.item_core = ItemMin(min_interactions, **kwargs)
        self.rounds = rounds

    def __call__(self, dataset: FrameT) -> FrameT:
        """Apply k-core filtering for the specified number of rounds.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset after applying k-core filtering for the specified rounds.
        """
        for _ in range(self.rounds):
            start_len = dataset.select(nw.len()).item()
            
            dataset = self.user_core(dataset)
            dataset = self.item_core(dataset)
            
            end_len = dataset.select(nw.len()).item()

            if end_len == start_len:
                break
        return dataset


@filter_registry.register("UserHeadN")
class UserHeadN(Filter):
    """Filter to keep only the first N interactions for each user,
    based on the timestamp.

    Args:
        num_interactions (int): Number of first interactions to keep for each user.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ValueError: If the provided argument is invalid.
    """

    def __init__(self, num_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if num_interactions <= 0:
            raise ValueError("num_interactions must be a positive integer.")
        self.num_interactions = num_interactions

    def __call__(self, dataset: FrameT) -> FrameT:
        """Select the first num_interactions for each user.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only the first num_interactions for each user.
        """
        # Check if timestamp is available
        is_timestamp_available = self.timestamp_label in dataset.columns

        if self.timestamp_label in dataset.columns:
            sort_col = nw.col(self.timestamp_label)
        else:
            dataset = dataset.with_row_index(name="__row_idx__")
            sort_col = nw.col("__row_idx__")

        # Calculate cumulative count per user (starts at 0)
        # Filter where count < N
        dataset = dataset.filter(
            sort_col.rank(method="ordinal").over(self.user_label) <= self.num_interactions
        )

        if "__row_idx__" in dataset.columns:
            dataset = dataset.drop("__row_idx__")

        return dataset


@filter_registry.register("UserTailN")
class UserTailN(Filter):
    """Filter to keep only the last N interactions for each user,
    based on the timestamp.

    Args:
        num_interactions (int): Number of last interactions to keep for each user.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ValueError: If the provided argument is invalid.
    """

    def __init__(self, num_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if num_interactions <= 0:
            raise ValueError("num_interactions must be a positive integer.")
        self.num_interactions = num_interactions

    def __call__(self, dataset: FrameT) -> FrameT:
        """Select the last num_interactions for each user.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only the last
                       num_interactions for each user.
        """
        is_timestamp_available = self.timestamp_label in dataset.columns

        if is_timestamp_available:
            sort_col = nw.col(self.timestamp_label)
        else:
            dataset = dataset.with_row_index(name="__row_idx__")
            sort_col = nw.col("__row_idx__")
        # Logic: Keep if row_index >= (total_count - N)
        # cum_count goes 0..Total-1
        
        dataset = dataset.filter(
            sort_col.rank(method="ordinal", descending=True).over(self.user_label) <= self.num_interactions
        )
        

        if "__row_idx__" in dataset.columns:
            dataset = dataset.drop("__row_idx__")

        return dataset

@filter_registry.register("DropUser")
class DropUser(Filter):
    """Filter to exclude one or a list of user IDs from the dataset.

    Args:
        user_ids_to_filter (Any | List[Any]): A single user ID or a list of user IDs to filter out.
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(self, user_ids_to_filter: Any | List[Any], **kwargs: Any):
        super().__init__(**kwargs)
        # Convert to list if a single user ID is provided
        if not isinstance(user_ids_to_filter, list):
            user_ids_to_filter = [user_ids_to_filter]
        self.user_ids_to_filter = user_ids_to_filter

    def __call__(self, dataset: FrameT) -> FrameT:
        """Exclude rows corresponding to the specified user IDs.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset without the specified users.
        """
        return dataset.filter(~nw.col(self.user_label).is_in(self.user_ids_to_filter))

@filter_registry.register("DropItem")
class DropItem(Filter):
    """Filter to exclude one or a list of item IDs from the dataset.

    Args:
        item_ids_to_filter (Any | List[Any]): A single item ID or a list of item IDs to filter out.
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(self, item_ids_to_filter: Any | List[Any], **kwargs: Any):
        super().__init__(**kwargs)
        # Convert to list if a single item ID is provided
        if not isinstance(item_ids_to_filter, list):
            item_ids_to_filter = [item_ids_to_filter]
        self.item_ids_to_filter = item_ids_to_filter

    def __call__(self, dataset: FrameT) -> FrameT:
        """Exclude rows corresponding to the specified item IDs.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset without the specified items.
        """
        return dataset.filter(~nw.col(self.item_label).is_in(self.item_ids_to_filter))

def apply_filtering(dataset: FrameT, filters: List[Filter]) -> FrameT:
    """Apply a list of filters to the dataset.

    Args:
        dataset (DataFrame): The dataset to filter.
        filters (List[Filter]): List of filters to apply.

    Returns:
        DataFrame: The filtered dataset after applying all filters.

    Raises:
        ValueError: If the dataset becomes empty after applying any filter.
    """
    if len(filters) == 0:
        logger.attention("No filters provided. Returning the original dataset.")
        return dataset

    initial_len = dataset.select(nw.len()).item()
    logger.msg(f"Applying filters to the dataset. Initial dataset size: {initial_len} rows.")
    start = time.time()

    for i, single_filter in enumerate(filters):
        dataset = single_filter(dataset)
        
        # Check if empty
        current_len = dataset.select(nw.len()).item()

        # Check if the dataset post filtering is empty
        if current_len == 0:
            raise ValueError(
                f"Dataset is empty after applying filter {i + 1}/{len(filters)}: "
                f"{single_filter.__class__.__name__}. Please check the filtering criteria."
            )

        logger.stats(
            f"After filter {i + 1}/{len(filters)} ({single_filter.__class__.__name__}): "
            f"{current_len} rows"
        )

    logger.positive(
        f"Filtering process completed. Final dataset size after filtering: {current_len}. "
        f"Total filtering time: {time.time() - start:.2f} seconds."
    )
    return dataset
