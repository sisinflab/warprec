from typing import List
from abc import ABC

import time
from pandas import DataFrame

from warprec.utils.logger import logger
from warprec.utils.registry import filter_registry


class Filter(ABC):
    """Abstract definition of a filter.
    Filters are used to process datasets by applying specific conditions
    or transformations to the data.
    """

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Apply the filter to the dataset."""
        raise NotImplementedError("Subclasses should implement this method.")


@filter_registry.register("MinRating")
class MinRating(Filter):
    """Filter to select rows based on a minimum rating.

    Args:
        min_rating (float): The minimum rating threshold.
    """

    def __init__(self, min_rating: float):
        self.min_rating = min_rating

    """Filter to select rows based on a rating condition."""

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select rows where the 'rating' column is greater than or equal to min_rating.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only rows with 'rating' >= min_rating.
        """
        return dataset[dataset["rating"] >= self.min_rating]


@filter_registry.register("UserAverage")
class UserAverage(Filter):
    """Filter to select users based on their average rating."""

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select rows where the 'rating' column is greater than the user average.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only rows with 'rating' > user average.
        """
        user_avg = dataset.groupby("user_id")["rating"].transform("mean")
        return dataset[dataset["rating"] > user_avg]


@filter_registry.register("UserMin")
class UserMin(Filter):
    """Filter to select users based on a minimum number of interactions.

    Args:
        min_interactions (int): Minimum number of interactions per user.
    """

    def __init__(self, min_interactions: int):
        self.min_interactions = min_interactions

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select users with at least min_interactions.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only users with interactions >= min_interactions.
        """
        user_counts = dataset["user_id"].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        return dataset[dataset["user_id"].isin(valid_users)]


@filter_registry.register("ItemMin")
class ItemMin(Filter):
    """Filter to select items based on a minimum number of interactions.

    Args:
        min_interactions (int): Minimum number of interactions per item.
    """

    def __init__(self, min_interactions: int):
        self.min_interactions = min_interactions

    def __call__(self, dataset: DataFrame) -> DataFrame:
        """Select items with at least min_interactions.

        Args:
            dataset (DataFrame): The dataset to filter.

        Returns:
            DataFrame: Filtered dataset containing only items with interactions >= min_interactions.
        """
        item_counts = dataset["item_id"].value_counts()
        valid_items = item_counts[item_counts >= self.min_interactions].index
        return dataset[dataset["item_id"].isin(valid_items)]


@filter_registry.register("IterativeKCore")
class IterativeKCore(Filter):
    """Iteratively apply k-core filtering to the dataset.

    Args:
        min_interactions (int): Minimum number of interactions for users/items.
    """

    def __init__(self, min_interactions: int):
        self.user_core = UserMin(min_interactions)
        self.item_core = ItemMin(min_interactions)

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
    """

    def __init__(self, rounds: int, min_interactions: int):
        self.user_core = UserMin(min_interactions)
        self.item_core = ItemMin(min_interactions)
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


def apply_filtering(dataset: DataFrame, filters: List[Filter]) -> DataFrame:
    """Apply a list of filters to the dataset.

    Args:
        dataset (DataFrame): The dataset to filter.
        filters (List[Filter]): List of filters to apply.

    Returns:
        DataFrame: The filtered dataset after applying all filters.
    """
    if len(filters) == 0:
        logger.attention("No filters provided. Returning the original dataset.")
        return dataset

    logger.msg(f"Applying filters to the dataset. Initial dataset size: {len(dataset)}")
    start = time.time()

    for i, filter in enumerate(filters):
        dataset = filter(dataset)
        logger.stats(
            f"After filter {i + 1}/{len(filters)} ({filter.__class__.__name__}): {len(dataset)} rows"
        )

    logger.positive(
        f"Filtering process completed. Final dataset size after filtering: {len(dataset)}. Total filtering time: {time.time() - start:.2f} seconds."
    )
    return dataset
