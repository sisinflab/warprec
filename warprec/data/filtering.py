# pylint: disable = too-few-public-methods
from typing import List, Any
from abc import ABC

import time
from pandas import DataFrame

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

    def __call__(self, dataset: DataFrame) -> DataFrame:
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

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select rows where the 'rating' column is greater than or equal to min_rating.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only rows with 'rating' >= min_rating.
        """
        return dataset[dataset[self.rating_label] >= self.min_rating]


@filter_registry.register("UserAverage")
class UserAverage(Filter):
    """Filter to select users based on their average rating.

    Args:
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select rows where the 'rating' column is greater than the user average.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only rows with 'rating' > user average.
        """
        user_avg = dataset.groupby(self.user_label)[self.rating_label].transform("mean")
        return dataset[dataset[self.rating_label] > user_avg]


@filter_registry.register("ItemAverage")
class ItemAverage(Filter):
    """Filter to select interactions for an item based on the item's average rating.

    Args:
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select rows where the 'rating' column is greater than the item average.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only rows with 'rating' > item average.
        """
        item_avg = dataset.groupby(self.item_label)[self.rating_label].transform("mean")
        return dataset[dataset[self.rating_label] > item_avg]


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

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select users with at least min_interactions.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only users with interactions >= min_interactions.
        """
        user_counts = dataset[self.user_label].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        return dataset[dataset[self.user_label].isin(valid_users)]


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

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select users with at most max_interactions.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only users with interactions <= max_interactions.
        """
        user_counts = dataset[self.user_label].value_counts()
        valid_users = user_counts[user_counts <= self.max_interactions].index
        return dataset[dataset[self.user_label].isin(valid_users)]


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

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select items with at least min_interactions.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only items with interactions >= min_interactions.
        """
        item_counts = dataset[self.item_label].value_counts()
        valid_items = item_counts[item_counts >= self.min_interactions].index
        return dataset[dataset[self.item_label].isin(valid_items)]


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

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select items with at most max_interactions.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only items with interactions <= max_interactions.
        """
        item_counts = dataset[self.item_label].value_counts()
        valid_items = item_counts[item_counts <= self.max_interactions].index
        return dataset[dataset[self.item_label].isin(valid_items)]


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

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Apply k-core filtering iteratively until no more users or items can be removed.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset after applying k-core filtering.
        """
        while True:
            filtered_dataset = self.user_core(dataset)
            filtered_dataset = self.item_core(filtered_dataset)

            if len(filtered_dataset) == len(dataset):
                break
            dataset = filtered_dataset

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

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Apply k-core filtering for the specified number of rounds.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset after applying k-core filtering for the specified rounds.
        """
        for _ in range(self.rounds):
            filtered_dataset = self.user_core(dataset)
            filtered_dataset = self.item_core(filtered_dataset)

            if len(filtered_dataset) == len(dataset):
                break
            dataset = filtered_dataset
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

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select the first num_interactions for each user.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only the first num_interactions for each user.
        """
        # Check if timestamp is available
        sorting_column = self.timestamp_label
        is_timestamp_available = self.timestamp_label in dataset.columns

        if not is_timestamp_available:
            # Fallback: Use original ordering
            sorting_column = "__ORIGINAL_ROW_ORDER__"
            dataset[sorting_column] = dataset.index

        sorted_dataset = dataset.sort_values(
            by=[self.user_label, sorting_column], ascending=[True, True]
        )
        filtered_dataset = sorted_dataset.groupby(self.user_label).head(
            self.num_interactions
        )

        # Remove temporary column if used
        if not is_timestamp_available:
            filtered_dataset.drop(columns=[sorting_column], inplace=True)

        return filtered_dataset


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

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select the last num_interactions for each user.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only the last
                       num_interactions for each user.
        """
        # Check if timestamp is available
        sorting_column = self.timestamp_label
        is_timestamp_available = self.timestamp_label in dataset.columns

        if not is_timestamp_available:
            # Fallback: Use original ordering
            sorting_column = "__ORIGINAL_ROW_ORDER__"
            dataset[sorting_column] = dataset.index

        sorted_dataset = dataset.sort_values(
            by=[self.user_label, sorting_column], ascending=[True, True]
        )
        filtered_dataset = sorted_dataset.groupby(self.user_label).tail(
            self.num_interactions
        )

        # Remove temporary column if used
        if not is_timestamp_available:
            filtered_dataset.drop(columns=[sorting_column], inplace=True)

        return filtered_dataset


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

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Exclude rows corresponding to the specified user IDs.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset without the specified users.
        """
        return dataset[~dataset[self.user_label].isin(self.user_ids_to_filter)]


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

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Exclude rows corresponding to the specified item IDs.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset without the specified items.
        """
        return dataset[~dataset[self.item_label].isin(self.item_ids_to_filter)]


def apply_filtering(dataset: DataFrame, filters: List[Filter]) -> DataFrame:
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

    logger.msg(f"Applying filters to the dataset. Initial dataset size: {len(dataset)}")
    start = time.time()

    for i, single_filter in enumerate(filters):
        dataset = single_filter(dataset).reset_index(drop=True)

        # Check if the dataset post filtering is empty
        if dataset.empty:
            raise ValueError(
                f"Dataset is empty after applying filter {i + 1}/{len(filters)}: "
                f"{single_filter.__class__.__name__}. Please check the filtering criteria."
            )

        logger.stats(
            f"After filter {i + 1}/{len(filters)} ({single_filter.__class__.__name__}): "
            f"{len(dataset)} rows"
        )

    logger.positive(
        f"Filtering process completed. Final dataset size after filtering: {len(dataset)}. "
        f"Total filtering time: {time.time() - start:.2f} seconds."
    )
    return dataset
