# pylint: disable=too-few-public-methods
from typing import Tuple, List, Any, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame
from elliotwo.utils.enums import SplittingStrategies
from elliotwo.utils.registry import splitting_registry


class SplittingStrategy(ABC):
    """Abstract definition of a splitting strategy."""

    @abstractmethod
    def split(
        self, data: DataFrame, **kwargs: Any
    ) -> Tuple[List[int], List[int], Optional[List[int]]]:
        """This method will split the data in train/test/validation splits.

        If the validation split was not set then it will be None.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], Optional[List[int]]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the test set.
                Optional[List[int]]: List of indexes that will end up in the validation set.
        """


@splitting_registry.register(SplittingStrategies.TEMPORAL_HOLDOUT)
class TemporalHoldoutSplit(SplittingStrategy):
    """The definition of a temporal holdout splitting strategy.

    Timestamp must be provided to use this strategy.
    """

    def split(
        self,
        data: DataFrame,
        test_ratio: float = 0.2,
        val_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[List[int], List[int], Optional[List[int]]]:
        """Implementation of the temporal holdout splitting. Original data will be splitted
        according to timestamp. If a seed has been set, the split will be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_ratio (float): The test set size.
            val_ratio (Optional[float]): The validation set size.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], Optional[List[int]]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the test set.
                Optional[List[int]]: List of indexes that will end up in the validation set.
        """
        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._temp_split(data, test_size=test_ratio)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if val_ratio:
            train_idxs, val_idxs = self._temp_split(
                data.iloc[train_idxs], test_size=val_ratio / (1 - test_ratio)
            )

        # Otherwise return train/test indices
        return train_idxs, test_idxs, val_idxs

    def _temp_split(
        self, data: DataFrame, test_size: float = 0.2
    ) -> Tuple[List[int], List[int]]:
        """Method to split data in two partitions, using a timestamp.

        This method will split data based on time, using as test
        samples the more recent transactions.

        Args:
            data (DataFrame): The original data in DataFrame format.
            test_size (float): Percentage of data that will end up in the second partition.

        Returns:
            Tuple[List[int], List[int]]:
                List[int]: List of indexes of the first partition.
                List[int]: List of indexes of the second partition.
        """
        # Set user and time label
        user_label = data.columns[0]
        time_label = data.columns[-1]

        # Single sorting by user and timestamp
        data = data.sort_values(by=[user_label, time_label])

        # Calculate index where to split
        user_counts = data[user_label].value_counts().sort_index()
        split_indices: DataFrame = np.floor(user_counts * (1 - test_size)).astype(int)
        split_indices[split_indices == 0] = 1  # Ensure at least one element in train

        # Generate a mask to efficiently split data
        split_mask = (
            data.groupby(user_label).cumcount()
            < split_indices.loc[data[user_label]].values
        )

        # Splitting
        train_idxs = data.index[split_mask].tolist()
        test_idxs = data.index[~split_mask].tolist()

        return train_idxs, test_idxs


@splitting_registry.register(SplittingStrategies.TEMPORAL_LEAVE_K_OUT)
class TemporalLeaveKOutSplit(SplittingStrategy):
    """The definition of a temporal leave k out splitting strategy.

    Timestamp must be provided to use this strategy.
    """

    def split(
        self,
        data: DataFrame,
        test_k: int = 5,
        val_k: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[List[int], List[int], Optional[List[int]]]:
        """Implementation of the temporal leave k out splitting. Original data will be splitted
        according to timestamp. If a seed has been set, the split will be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_k (int): The test set k value.
            val_k (Optional[int]): The validation set k value.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], Optional[List[int]]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the test set.
                Optional[List[int]]: List of indexes that will end up in the validation set.
        """
        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._temp_split(data, test_k=test_k)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if val_k:
            train_idxs, val_idxs = self._temp_split(data.iloc[train_idxs], test_k=val_k)

        # Otherwise return train/test indices
        return train_idxs, test_idxs, val_idxs

    def _temp_split(
        self, data: DataFrame, test_k: int = 1
    ) -> Tuple[List[int], List[int]]:
        """Method to split data in two partitions, using a timestamp.

        This method will split data based on time, using as test
        samples the more recent transactions.

        Args:
            data (DataFrame): The original data in DataFrame format.
            test_k (int): Number of transaction that will end up in the second partition.

        Returns:
            Tuple[List[int], List[int]]:
                List[int]: List of indexes of the first partition.
                List[int]: List of indexes of the second partition.
        """
        # Set user and time label
        user_label = data.columns[0]  # Assuming first column is user ID
        time_label = data.columns[-1]  # Assuming last column is timestamp

        # Single sorting by user and timestamp
        data = data.sort_values(by=[user_label, time_label])

        # Determine the split indices for each user
        user_counts = data[user_label].value_counts().sort_index()
        valid_users = user_counts[user_counts > test_k].index  # Users with enough data

        # Filter out users with insufficient transactions
        data = data[data[user_label].isin(valid_users)]

        # Determine the split indices for each user
        split_indices = user_counts - test_k

        # Create a split mask: True for training, False for test
        split_mask = (
            data.groupby(user_label).cumcount()
            < split_indices.loc[data[user_label]].values
        )

        # Splitting based on the mask
        train_idxs = data.index[split_mask].tolist()
        test_idxs = data.index[~split_mask].tolist()

        return train_idxs, test_idxs


@splitting_registry.register(SplittingStrategies.TIMESTAMP_SLICING)
class TimestampSplit(SplittingStrategy):
    """Splits data based on a timestamp. Either a fixed timestamp or a
    'best' timestamp can be used.

    In case of best timestamp strategy, the slicing will be conducted finding the
    timestamp that better separates the transactions. Using the normal fixed strategy
    some users might be completely cut out of the train or the test. With the best
    strategy we ensure that the most amount of users will be represented in both sets.

    For further details about the 'best' timestamp, check the `paper <https://link.springer.com/chapter/10.1007/978-3-030-15712-8_63>`_.

    Timestamp must be provided to use this strategy.
    """

    def split(
        self,
        data: DataFrame,
        timestamp: Union[int, str] = 0,
        **kwargs: Any,
    ) -> Tuple[List[int], List[int], Optional[List[int]]]:
        """Implementation of the fixed timestamp splitting.

        Args:
            data (DataFrame): The DataFrame to be split.
            timestamp (Union[int, str]): The timestamp to split data for test set.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], Optional[List[int]]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the test set.
                Optional[List[int]]: List of indexes that will end up in the validation set.
        """
        if timestamp == "best":
            best_timestamp = self._best_split(data)
            train_idxs, test_idxs = self._fixed_split(data, best_timestamp)
        else:
            train_idxs, test_idxs = self._fixed_split(data, int(timestamp))

        # Return train/test indices
        return train_idxs, test_idxs, None

    def _best_split(
        self, data: DataFrame, min_below: int = 1, min_over: int = 1
    ) -> int:
        """Optimized method to find the best split timestamp for partitioning data.

        Args:
            data (DataFrame): The original data in DataFrame format.
            min_below (int): Minimum number of transactions below the timestamp.
            min_over (int): Minimum number of transactions above the timestamp.

        Returns:
            int: Best timestamp for splitting user transactions.
        """
        user_label = data.columns[0]  # Assuming first column is user ID
        time_label = data.columns[-1]  # Assuming last column is timestamp

        unique_timestamps = np.sort(data[time_label].unique())
        n_candidates = unique_timestamps.shape[0]

        candidate_scores = np.zeros(n_candidates, dtype=int)

        user_groups = data.groupby(user_label)

        for user, group in user_groups:
            user_ts = np.sort(group[time_label].values)
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
        self, data: DataFrame, timestamp: int
    ) -> Tuple[List[int], List[int]]:
        """Method to split data in two partitions, using a fixed timestamp.

        This method will split data based on the timestamp provided.

        Args:
            data (DataFrame): The original data in DataFrame format.
            timestamp (int): The timestamp to be used for splitting.

        Returns:
            Tuple[List[int], List[int]]:
                List[int]: List of indexes of the first partition.
                List[int]: List of indexes of the second partition.
        """
        # Set time label
        time_label = data.columns[-1]  # Assuming last column is timestamp

        # Create a boolean mask for the split
        split_mask = data[time_label] < timestamp  # True for train, False for test

        # Splitting
        train_idxs = data.index[split_mask].tolist()
        test_idxs = data.index[~split_mask].tolist()

        return train_idxs, test_idxs


@splitting_registry.register(SplittingStrategies.RANDOM_RATIO)
class RandomRatioSplit(SplittingStrategy):
    """The definition of a random ratio splitting strategy."""

    def split(
        self,
        data: DataFrame,
        test_ratio: float = 0.2,
        val_ratio: Optional[float] = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> Tuple[List[int], List[int], Optional[List[int]]]:
        """Implementation of the random ratio splitting.
        If a seed has been set, the split will be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_ratio (float): The test set size.
            val_ratio (Optional[float]): The validation set size.
            seed (int): The seed for the random number generator.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], Optional[List[int]]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the test set.
                Optional[List[int]]: List of indexes that will end up in the validation set.
        """
        np.random.seed(seed)

        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._random_ratio(data, test_size=test_ratio)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if val_ratio:
            train_idxs, val_idxs = self._random_ratio(
                data.iloc[train_idxs], test_size=val_ratio / (1 - test_ratio)
            )

        # Otherwise return train/test indices
        return train_idxs, test_idxs, val_idxs

    def _random_ratio(
        self, data: DataFrame, test_size: float = 0.2
    ) -> Tuple[List[int], List[int]]:
        """Method to split data in two partitions, using a ratio.

        This method will split data based on the ratio provided.

        Args:
            data (DataFrame): The original data in DataFrame format.
            test_size (float): Percentage of data that will end up in the second partition.

        Returns:
            Tuple[List[int], List[int]]:
                List[int]: List of indexes of the first partition.
                List[int]: List of indexes of the second partition.
        """
        all_indices = data.index.to_numpy()
        permuted = np.random.permutation(all_indices)
        n_total = len(permuted)
        n_test = int(n_total * test_size)

        test_idxs = permuted[:n_test].tolist()
        train_idxs = permuted[n_test:].tolist()

        return train_idxs, test_idxs  # type: ignore[return-value]


@splitting_registry.register(SplittingStrategies.RANDOM_LEAVE_K_OUT)
class RandomLeaveKOutSplit(SplittingStrategy):
    """The definition of a random leave k out splitting strategy."""

    def split(
        self,
        data: DataFrame,
        test_k: int = 5,
        val_k: Optional[int] = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> Tuple[List[int], List[int], Optional[List[int]]]:
        """Implementation of the random leave k out splitting.
        If a seed has been set, the split will be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_k (int): The test set k.
            val_k (Optional[int]): The validation set k.
            seed (int): The seed for the random number generator.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], Optional[List[int]]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the test set.
                Optional[List[int]]: List of indexes that will end up in the validation set.
        """
        np.random.seed(seed)

        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._random_leave(data, test_k=test_k)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if val_k:
            train_idxs, val_idxs = self._random_leave(
                data.iloc[train_idxs], test_k=val_k
            )

        # Otherwise return train/test indices
        return train_idxs, test_idxs, val_idxs

    def _random_leave(
        self, data: DataFrame, test_k: int = 1
    ) -> Tuple[List[int], List[int]]:
        """Method to split data in two partitions, using a ratio.

        This method will split data based on the ratio provided.

        Args:
            data (DataFrame): The original data in DataFrame format.
            test_k (int): Number of transaction that will end up in the second partition.
        Returns:
            Tuple[List[int], List[int]]:
                List[int]: List of indexes of the first partition.
                List[int]: List of indexes of the second partition.
        """
        user_label = data.columns[0]  # Assuming first column is the user ID
        train_all = []  # type: ignore[var-annotated]
        test_all = []  # type: ignore[var-annotated]

        # Process each user group separately
        for user, group in data.groupby(user_label):
            indices = group.index.tolist()
            # Proceed only if the user has more than test_k interactions
            if len(indices) <= test_k:
                continue

            # Shuffle the indices for the current user
            permuted = np.random.permutation(indices).tolist()

            # Select first test_k indices for the second partition and the rest for the first partition
            user_test = permuted[:test_k]
            user_train = permuted[test_k:]

            train_all.extend(user_train)
            test_all.extend(user_test)

        return train_all, test_all
