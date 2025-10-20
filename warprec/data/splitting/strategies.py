# pylint: disable=too-few-public-methods
# mypy: disable-error-code=override
from typing import Tuple, Any, Union, List
from abc import ABC

import numpy as np
import pandas as pd
from pandas import DataFrame
from warprec.utils.enums import SplittingStrategies
from warprec.utils.registry import splitting_registry


class SplittingStrategy(ABC):
    """Abstract definition of a splitting strategy."""

    def __call__(
        self, data: DataFrame, **kwargs: Any
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """This method will split the data in train/test/validation splits.

        If the validation split was not set then it will be None.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame, DataFrame]]:
                - DataFrame: First partition of splitted data.
                - DataFrame: Second partition of splitted data.
        """


@splitting_registry.register(SplittingStrategies.TEMPORAL_HOLDOUT)
class TemporalHoldoutSplit(SplittingStrategy):
    """The definition of a temporal holdout splitting strategy.

    In case the timestamp will not be provided, the former order
    of the transactions will be used.
    """

    def __call__(
        self,
        data: DataFrame,
        user_id_label: str = "user_id",
        timestamp_label: str = "timestamp",
        ratio: float = 0.2,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """Method to split data in two partitions, using a timestamp or the
        original order.

        This method will split data based on time, using as test
        samples the more recent transactions.

        Args:
            data (DataFrame): The original data in DataFrame format.
            user_id_label (str): The user_id label.
            timestamp_label (str): The timestamp label.
            ratio (float): Percentage of data that will end up in the second partition.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame, DataFrame]]:
                - DataFrame: First partition of splitted data.
                - DataFrame: Second partition of splitted data.
        """
        if timestamp_label in data.columns:
            # Single sorting by user and timestamp
            data = data.sort_values(by=[user_id_label, timestamp_label])

        # Calculate index where to split
        user_counts = data[user_id_label].value_counts().sort_index()
        split_indices: DataFrame = np.floor(user_counts * (1 - ratio)).astype(int)
        split_indices[split_indices == 0] = 1  # Ensure at least one element in train

        # Generate a mask to efficiently split data
        split_mask = (
            data.groupby(user_id_label).cumcount()
            < split_indices.loc[data[user_id_label]].values
        )

        return [(data[split_mask], data[~split_mask])]


@splitting_registry.register(SplittingStrategies.TEMPORAL_LEAVE_K_OUT)
class TemporalLeaveKOutSplit(SplittingStrategy):
    """The definition of a temporal leave k out splitting strategy.

    In case the timestamp will not be provided, the former order
    of the transactions will be used.
    """

    def __call__(
        self,
        data: DataFrame,
        user_id_label: str = "user_id",
        timestamp_label: str = "timestamp",
        k: int = 1,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """Method to split data in two partitions, using a timestamp or the
        original order.

        This method will split data based on time, using as test
        samples the more recent transactions.

        Args:
            data (DataFrame): The original data in DataFrame format.
            user_id_label (str): The user_id label.
            timestamp_label (str): The timestamp label.
            k (int): Number of transaction that will end up in the second partition.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame, DataFrame]]:
                - DataFrame: First partition of splitted data.
                - DataFrame: Second partition of splitted data.
        """
        if timestamp_label in data.columns:
            # Single sorting by user and timestamp
            data = data.sort_values(by=[user_id_label, timestamp_label])

        # Determine the split indices for each user
        user_counts = data[user_id_label].value_counts().sort_index()
        valid_users = user_counts[user_counts > k].index  # Users with enough data

        # Filter out users with insufficient transactions
        data = data[data[user_id_label].isin(valid_users)]

        # Determine the split indices for each user
        split_indices = user_counts - k

        # Create a split mask: True for training, False for test
        split_mask = (
            data.groupby(user_id_label).cumcount()
            < split_indices.loc[data[user_id_label]].values
        )

        return [(data[split_mask], data[~split_mask])]


@splitting_registry.register(SplittingStrategies.TIMESTAMP_SLICING)
class TimestampSlicingSplit(SplittingStrategy):
    """Splits data based on a timestamp. Either a fixed timestamp or a
    'best' timestamp can be used.

    In case of best timestamp strategy, the slicing will be conducted finding the
    timestamp that better separates the transactions. Using the normal fixed strategy
    some users might be completely cut out of the train or the test. With the best
    strategy we ensure that the most amount of users will be represented in both sets.

    For further details about the 'best' timestamp, check the `paper <https://link.springer.com/chapter/10.1007/978-3-030-15712-8_63>`_.

    Timestamp must be provided to use this strategy.
    """

    def __call__(
        self,
        data: DataFrame,
        user_id_label: str = "user_id",
        timestamp_label: str = "timestamp",
        timestamp: Union[int, str] = 0,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """Implementation of the fixed timestamp splitting.

        Args:
            data (DataFrame): The DataFrame to be split.
            user_id_label (str): The user_id label.
            timestamp_label (str): The timestamp label.
            timestamp (Union[int, str]): The timestamp to split data for test set.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame, DataFrame]]:
                - DataFrame: First partition of splitted data.
                - DataFrame: Second partition of splitted data.
        """
        if timestamp == "best":
            best_timestamp = self._best_split(data, user_id_label, timestamp_label)
            first_partition, second_partition = self._fixed_split(
                data, best_timestamp, timestamp_label
            )
        else:
            first_partition, second_partition = self._fixed_split(
                data, int(timestamp), timestamp_label
            )

        return [(first_partition, second_partition)]

    def _best_split(
        self,
        data: DataFrame,
        user_id_label: str = "user_id",
        timestamp_label: str = "timestamp",
        min_below: int = 1,
        min_over: int = 1,
    ) -> int:
        """Optimized method to find the best split timestamp for partitioning data.

        Args:
            data (DataFrame): The original data in DataFrame format.
            user_id_label (str): The user_id label.
            timestamp_label (str): The timestamp label.
            min_below (int): Minimum number of transactions below the timestamp.
            min_over (int): Minimum number of transactions above the timestamp.

        Returns:
            int: Best timestamp for splitting user transactions.
        """
        unique_timestamps = np.sort(data[timestamp_label].unique())
        n_candidates = unique_timestamps.shape[0]

        candidate_scores = np.zeros(n_candidates, dtype=int)

        user_groups = data.groupby(user_id_label)

        for _, group in user_groups:
            user_ts = np.sort(group[timestamp_label].values)
            total_events = user_ts.shape[0]
            below_counts = np.searchsorted(user_ts, unique_timestamps, side="left")
            over_counts = total_events - below_counts

            valid = (below_counts >= min_below) & (over_counts >= min_over)
            candidate_scores += valid.astype(int)

        max_score = candidate_scores.max()

        valid_candidates = unique_timestamps[candidate_scores == max_score]
        best_timestamp = valid_candidates.max()

        return best_timestamp

    def _fixed_split(
        self, data: DataFrame, timestamp: int, timestamp_label: str = "timestamp"
    ) -> Tuple[DataFrame, DataFrame]:
        """Method to split data in two partitions, using a fixed timestamp.

        This method will split data based on the timestamp provided.

        Args:
            data (DataFrame): The original data in DataFrame format.
            timestamp (int): The timestamp to be used for splitting.
            timestamp_label (str): The timestamp label.

        Returns:
            Tuple[DataFrame, DataFrame]:
                - DataFrame: First partition of splitted data.
                - DataFrame: Second partition of splitted data.
        """
        # Create a boolean mask for the split
        split_mask = data[timestamp_label] < timestamp  # True for train, False for test

        return data[split_mask], data[~split_mask]


@splitting_registry.register(SplittingStrategies.RANDOM_HOLDOUT)
class RandomHoldoutSplit(SplittingStrategy):
    """The definition of a random ratio splitting strategy."""

    def __call__(
        self,
        data: DataFrame,
        ratio: float = 0.2,
        seed: int = 42,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """Method to split data in two partitions, using a ratio.

        This method will split data based on the ratio provided.

        Args:
            data (DataFrame): The original data in DataFrame format.
            ratio (float): Percentage of data that will end up in the second partition.
            seed (int): The seed used for the random number generator.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame, DataFrame]]:
                - DataFrame: First partition of splitted data.
                - DataFrame: Second partition of splitted data.
        """
        np.random.seed(seed)
        all_indices = data.index.to_numpy()
        permuted = np.random.permutation(all_indices)
        n_total = len(permuted)
        n_partition = int(n_total * ratio)

        first_partition_idxs = permuted[n_partition:]
        second_partition_idxs = permuted[:n_partition]

        return [(data.loc[first_partition_idxs], data.loc[second_partition_idxs])]  # type: ignore[return-value]


@splitting_registry.register(SplittingStrategies.RANDOM_LEAVE_K_OUT)
class RandomLeaveKOutSplit(SplittingStrategy):
    """The definition of a random leave k out splitting strategy."""

    def __call__(
        self,
        data: DataFrame,
        user_id_label: str = "user_id",
        k: int = 1,
        seed: int = 42,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """Method to split data in two partitions, using a ratio.

        This method will split data based on the ratio provided.

        Args:
            data (DataFrame): The original data in DataFrame format.
            user_id_label (str): The user_id label.
            k (int): Number of transaction that will end up in the second partition.
            seed (int): The seed used for the random number generator.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame, DataFrame]]:
                - DataFrame: First partition of splitted data.
                - DataFrame: Second partition of splitted data.
        """
        first_partition_all: list[Any] = []
        second_partition_all: list[Any] = []

        # Process each user group separately
        for _, group in data.groupby(user_id_label):
            indices = group.index.tolist()
            # Proceed only if the user has more than test_k interactions
            if len(indices) <= k:
                continue

            # Shuffle the indices for the current user
            permuted = np.random.permutation(indices).tolist()

            # Select first test_k indices for the second partition and the rest for the first partition
            first_partition_idxs = permuted[k:]
            second_partition_idxs = permuted[:k]

            first_partition_all.extend(first_partition_idxs)
            second_partition_all.extend(second_partition_idxs)

        return [(data.loc[first_partition_all], data.loc[second_partition_all])]


@splitting_registry.register(SplittingStrategies.K_FOLD_CROSS_VALIDATION)
class KFoldCrossValidation(SplittingStrategy):
    """The definition of KFold Cross Validation."""

    def __call__(
        self,
        data: DataFrame,
        folds: int,
        user_id_label: str = "user_id",
        **kwargs: Any,
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """Method to split data in 'folds' times.

        Args:
            data (DataFrame): The original data in DataFrame format.
            folds (int): The number of folds to create.
            user_id_label (str): The user_id label.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame, DataFrame]]:
                - DataFrame: First partition of splitted data.
                - DataFrame: Second partition of splitted data.
        """
        for _, group in data.groupby(user_id_label):
            data.loc[group.index, "fold"] = self.fold_list_generator(len(group), folds)

        data["fold"] = pd.to_numeric(data["fold"], downcast="integer")
        tuple_list = []
        for i in range(folds):
            test = data[data["fold"] == i].drop(columns=["fold"]).reset_index(drop=True)
            train = (
                data[data["fold"] != i].drop(columns=["fold"]).reset_index(drop=True)
            )
            tuple_list.append((train, test))
        return tuple_list

    def fold_list_generator(self, length: int, folds: int = 5) -> List[int]:
        """Utility method to create the fold column for transaction data.

        Args:
            length (int): The length of the list.
            folds (int): The number of folds to create.

        Returns:
            List[int]: The list of repetitive folds indices.
        """

        def infinite_looper(folds: int = 5):
            while True:
                for f in range(folds):
                    yield f

        looper = infinite_looper(folds)
        return [next(looper) for _ in range(length)]
