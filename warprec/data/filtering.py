import inspect
from typing import List, Any
from abc import ABC

import time
import narwhals as nw
from narwhals.typing import FrameT
from narwhals.dataframe import DataFrame

from warprec.errors import ConfigurationError, DataError
from warprec.utils.logger import logger
from warprec.utils.registry import filter_registry


class Filter(ABC):
    """Abstract definition of a filter.
    Filters are used to process datasets by applying specific conditions
    or transformations to the data.

    Args:
        user_id_label (str): Column name for user IDs.
        item_id_label (str): Column name for item IDs.
        rating_label (str): Column name for ratings.
        timestamp_label (str): Column name for timestamps.
        **kwargs (Any): The other column labels the configuration hands every
            filter. Any other key is refused.

    Raises:
        ConfigurationError: If a keyword is neither a parameter of the filter nor a
            column label.
    """

    # The labels a configuration passes to every filter besides the four above.
    _OTHER_LABELS = frozenset({"cluster_label", "context_labels"})

    def __init__(
        self,
        user_id_label: str = "user_id",
        item_id_label: str = "item_id",
        rating_label: str = "rating",
        timestamp_label: str = "timestamp",
        **kwargs: Any,
    ):
        # A misspelled parameter would otherwise land here and be ignored,
        # leaving the filter to run with its default, or not at all.
        unknown = sorted(set(kwargs) - self._OTHER_LABELS)
        if unknown:
            accepted = [
                name
                for name, param in inspect.signature(
                    type(self).__init__
                ).parameters.items()
                if name != "self"
                and param.kind
                not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            ]
            raise ConfigurationError(
                f"The filter {type(self).__name__} does not take "
                f"{', '.join(repr(key) for key in unknown)}. The parameters it "
                f"takes are {accepted or 'none'}."
            )

        self.user_label = user_id_label
        self.item_label = item_id_label
        self.rating_label = rating_label
        self.timestamp_label = timestamp_label

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Apply the filter to the dataset."""
        raise NotImplementedError("Subclasses should implement this method.")

    # The columns _order_in_time attaches: a row's 1-based place in its user's
    # temporal order, the user's number of rows, and the row's place in the input.
    _PLACE = "__temporal_place__"
    _COUNT = "__temporal_count__"
    _FILE = "__file_order__"

    def _order_in_time(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Sort each user's rows into temporal order and number them in it.

        Temporal order is the timestamp, then the position in the file: rows that
        share a timestamp keep the order they were read in, and the most recent
        row is the last one. Without a timestamp column, file order alone is used.
        The number is taken from the position in the sorted frame rather than
        from a rank of the timestamp: an ordinal rank gives a tie to the row that
        appears first whichever way it runs, so a descending one called the
        earlier of two same-second rows the most recent.

        Args:
            dataset (DataFrame[Any]): The dataset to order.

        Returns:
            DataFrame[Any]: The dataset sorted by user and temporal order, with
                the row's place in its user's order, the user's number of rows
                and the row's place in the input attached under ``_PLACE``,
                ``_COUNT`` and ``_FILE``. A filter sorts by ``_FILE`` before it
                hands the rows on, because what follows breaks ties by row order.
        """
        position = "__temporal_position__"
        by = [self.user_label, self._FILE]
        if self.timestamp_label in dataset.columns:
            by.insert(1, self.timestamp_label)

        ordered = (
            dataset.with_row_index(name=self._FILE)
            .sort(by=by)
            .with_row_index(name=position)
        )
        return ordered.with_columns(
            nw.col(position)
            .rank(method="ordinal")
            .over(self.user_label)
            .cast(nw.Int64)
            .alias(self._PLACE),
            nw.len().over(self.user_label).cast(nw.Int64).alias(self._COUNT),
        ).drop(position)


@filter_registry.register("MinRating")
class MinRating(Filter):
    """Filter to select rows based on a minimum rating.

    Args:
        min_rating (float): The minimum rating threshold.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ConfigurationError: If the provided argument is invalid.
    """

    def __init__(self, min_rating: float, **kwargs: Any):
        super().__init__(**kwargs)
        if min_rating <= 0:
            raise ConfigurationError("min_rating must be a positive float.")
        self.min_rating = min_rating

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Select rows where the 'rating' column is greater than or equal to min_rating.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset containing only rows with 'rating' >= min_rating.
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

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Select rows where the 'rating' column is greater than the user average.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset containing only rows with 'rating' > user average.
        """
        return dataset.filter(
            nw.col(self.rating_label)
            > nw.col(self.rating_label).mean().over(self.user_label)
        )


@filter_registry.register("ItemAverage")
class ItemAverage(Filter):
    """Filter to select interactions for an item based on the item's average rating.

    Args:
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Select rows where the 'rating' column is greater than the item average.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset containing only rows with 'rating' > item average.
        """
        return dataset.filter(
            nw.col(self.rating_label)
            > nw.col(self.rating_label).mean().over(self.item_label)
        )


@filter_registry.register("UserMin")
class UserMin(Filter):
    """Filter to select users based on a minimum number of interactions.

    Args:
        min_interactions (int): Minimum number of interactions per user.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ConfigurationError: If the provided argument is invalid.
    """

    def __init__(self, min_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if min_interactions <= 0:
            raise ConfigurationError("min_interactions must be a positive integer.")
        self.min_interactions = min_interactions

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Select users with at least min_interactions.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset containing only users with interactions >= min_interactions.
        """
        return dataset.filter(nw.len().over(self.user_label) >= self.min_interactions)


@filter_registry.register("UserMax")
class UserMax(Filter):
    """Filter to select users based on a maximum number of interactions.

    Args:
        max_interactions (int): Maximum number of interactions per user.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ConfigurationError: If the provided argument is invalid.
    """

    def __init__(self, max_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if max_interactions <= 0:
            raise ConfigurationError("max_interactions must be a positive integer.")
        self.max_interactions = max_interactions

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Select users with at most max_interactions.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset containing only users with interactions <= max_interactions.
        """
        return dataset.filter(nw.len().over(self.user_label) <= self.max_interactions)


@filter_registry.register("ItemMin")
class ItemMin(Filter):
    """Filter to select items based on a minimum number of interactions.

    Args:
        min_interactions (int): Minimum number of interactions per item.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ConfigurationError: If the provided argument is invalid.
    """

    def __init__(self, min_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if min_interactions <= 0:
            raise ConfigurationError("min_interactions must be a positive integer.")
        self.min_interactions = min_interactions

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Select items with at least min_interactions.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset containing only items with interactions >= min_interactions.
        """
        return dataset.filter(nw.len().over(self.item_label) >= self.min_interactions)


@filter_registry.register("ItemMax")
class ItemMax(Filter):
    """Filter to select items based on a maximum number of interactions.

    Args:
        max_interactions (int): Maximum number of interactions per item.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ConfigurationError: If the provided argument is invalid.
    """

    def __init__(self, max_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if max_interactions <= 0:
            raise ConfigurationError("max_interactions must be a positive integer.")
        self.max_interactions = max_interactions

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Select items with at most max_interactions.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset containing only items with interactions <= max_interactions.
        """
        return dataset.filter(nw.len().over(self.item_label) <= self.max_interactions)


@filter_registry.register("IterativeKCore")
class IterativeKCore(Filter):
    """Iteratively apply k-core filtering to the dataset.

    Args:
        min_interactions (int): Minimum number of interactions for users/items.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ConfigurationError: If the provided argument is invalid.
    """

    def __init__(self, min_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if min_interactions <= 0:
            raise ConfigurationError("min_interactions must be a positive integer.")
        self.user_core = UserMin(min_interactions, **kwargs)
        self.item_core = ItemMin(min_interactions, **kwargs)

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Apply k-core filtering iteratively until no more users or items can be removed.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset after applying k-core filtering.
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
        ConfigurationError: If the provided argument is invalid.
    """

    def __init__(self, rounds: int, min_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if rounds <= 0:
            raise ConfigurationError("rounds must be a positive integer.")
        if min_interactions <= 0:
            raise ConfigurationError("min_interactions must be a positive integer.")
        self.user_core = UserMin(min_interactions, **kwargs)
        self.item_core = ItemMin(min_interactions, **kwargs)
        self.rounds = rounds

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Apply k-core filtering for the specified number of rounds.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset after applying k-core filtering for the specified rounds.
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
    based on the timestamp. Interactions sharing a timestamp keep their
    order in the file.

    Args:
        num_interactions (int): Number of first interactions to keep for each user.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ConfigurationError: If the provided argument is invalid.
    """

    def __init__(self, num_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if num_interactions <= 0:
            raise ConfigurationError("num_interactions must be a positive integer.")
        self.num_interactions = num_interactions

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Select the first num_interactions for each user.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset containing only the first num_interactions for each user.
        """
        ordered = self._order_in_time(dataset)
        return (
            ordered.filter(nw.col(self._PLACE) <= self.num_interactions)
            .sort(self._FILE)
            .drop([self._PLACE, self._COUNT, self._FILE])
        )


@filter_registry.register("UserTailN")
class UserTailN(Filter):
    """Filter to keep only the last N interactions for each user,
    based on the timestamp. Interactions sharing a timestamp keep their
    order in the file, the later one being the more recent.

    Args:
        num_interactions (int): Number of last interactions to keep for each user.
        **kwargs (Any): Additional keyword arguments.

    Raises:
        ConfigurationError: If the provided argument is invalid.
    """

    def __init__(self, num_interactions: int, **kwargs: Any):
        super().__init__(**kwargs)
        if num_interactions <= 0:
            raise ConfigurationError("num_interactions must be a positive integer.")
        self.num_interactions = num_interactions

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Select the last num_interactions for each user.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset containing only the last
                       num_interactions for each user.
        """
        ordered = self._order_in_time(dataset)
        return (
            ordered.filter(
                nw.col(self._PLACE) > nw.col(self._COUNT) - self.num_interactions
            )
            .sort(self._FILE)
            .drop([self._PLACE, self._COUNT, self._FILE])
        )


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

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Exclude rows corresponding to the specified user IDs.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset without the specified users.
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

    def __call__(self, dataset: DataFrame[Any]) -> DataFrame[Any]:
        """Exclude rows corresponding to the specified item IDs.

        Args:
            dataset (DataFrame[Any]): The dataset to filter.

        Returns:
            DataFrame[Any]: Filtered dataset without the specified items.
        """
        return dataset.filter(~nw.col(self.item_label).is_in(self.item_ids_to_filter))


def apply_filtering(data: FrameT, filters: List[Filter]) -> DataFrame[Any]:
    """Apply a list of filters to the dataset.

    Args:
        data (FrameT): The dataset to filter.
        filters (List[Filter]): List of filters to apply.

    Returns:
        DataFrame[Any]: The filtered dataset after applying all filters.

    Raises:
        DataError: If the dataset becomes empty after applying any filter.
    """
    if isinstance(data, nw.LazyFrame):
        mat_data = data.collect()
    else:
        mat_data = data

    if len(filters) == 0:
        logger.attention("No filters provided. Returning the original dataset.")
        return mat_data

    initial_len = mat_data.select(nw.len()).item()
    logger.msg(
        f"Applying filters to the dataset. Initial dataset size: {initial_len} rows."
    )
    start = time.time()

    for i, single_filter in enumerate(filters):
        mat_data = single_filter(mat_data)

        # Check if empty
        current_len = mat_data.select(nw.len()).item()

        # Check if the dataset post filtering is empty
        if current_len == 0:
            raise DataError(
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
    return mat_data
