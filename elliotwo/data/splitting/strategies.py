# pylint: disable=too-few-public-methods
from typing import Tuple, List, Any, Optional
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
    ) -> Tuple[List[int], List[int], List[int]]:
        """This method will split the data in train/test/validation splits.

        If the validation split was not set then it will be None.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the test set.
                List[int]: List of indexes that will end up in the validation set.
        """


@splitting_registry.register(SplittingStrategies.RANDOM)
class RandomSplit(SplittingStrategy):
    """The definition of the random split strategy.

    This splitting will be executed randomly, unless a seed is set.
    """

    def split(
        self,
        data: DataFrame,
        test_ratio: float = 0.2,
        val_ratio: Optional[float] = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the random splitting. Original data will
        be splitted randomly. If a seed has been set, the split will be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_ratio (float): The size of the test set.
            val_ratio (Optional[float]): The size of the validation set.
            seed (int): The seed to use during the split.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the test set.
                List[int]: List of indexes that will end up in the validation set.
        """
        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._ratio_split(data, test_size=test_ratio, seed=seed)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if val_ratio:
            train_idxs, val_idxs = self._ratio_split(
                data.iloc[train_idxs], test_size=val_ratio / (1 - test_ratio), seed=seed
            )

        # Otherwise return train/test indices
        return train_idxs, test_idxs, val_idxs

    def _ratio_split(
        self, data: DataFrame, test_size: float = 0.2, seed: int = 42
    ) -> Tuple[List[int], List[int]]:
        """Method used to split a set of data into two partition,
        respecting the ratio given as input.

        The method used to split data is random. If a seed was set,
        then the split will be reproducible.

        Args:
            data (DataFrame): The original data in DataFrame format.
            test_size (float): This value represent the percentage of
                data that will be taken out.
            seed (int): The seed to use during the split.

        Returns:
            Tuple[List[int], List[int]]:
                List[int]: List of indexes of the first partition.
                List[int]: List of indexes of the second partition.

        TODO: This method does not ensures that every user is in the training set.
        """
        np.random.seed(seed)

        user_groups = data.groupby(
            "user_id"
        ).indices  # Dictionary {user_id: np.array(indices)}

        train_indices = []
        test_indices = []

        for _, indices in user_groups.items():
            if len(indices) == 1:
                # If a user has only one interaction, force it into train
                train_indices.append(indices[0])
            else:
                # Shuffle indices and split
                np.random.shuffle(indices)
                split_idx = int(len(indices) * (1 - test_size))
                train_indices.extend(indices[:split_idx])
                test_indices.extend(indices[split_idx:])

        return train_indices, test_indices


@splitting_registry.register(SplittingStrategies.LEAVE_ONE_OUT)
class LeaveOneOutSplit(SplittingStrategy):
    """The definition of a leave-one-out splitting strategy."""

    def split(
        self,
        data: DataFrame,
        test_k: int = 1,
        val_k: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the leave-one-out splitting. If a seed has
        been set, the split will be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_k (int): The test k.
            val_k (Optional[int]): The validation k.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the validation set.
                List[int]: List of indexes that will end up in the test set.
        """
        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._k_split(data, test_k)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if val_k:
            train_idxs, val_idxs = self._k_split(data.iloc[train_idxs], val_k)

        # Otherwise return train/test indices
        return train_idxs, test_idxs, val_idxs

    def _k_split(self, data: DataFrame, k: int = 1) -> Tuple[List[int], List[int]]:
        """Method to split data in two partitions, using a fixed number.

        This method will take in account some limit examples like
        users with less then k transactions.

        Args:
            data (DataFrame): The original data in DataFrame format.
            k (int): The number of elements to be included in the second partition.

        Returns:
            Tuple[List[int], List[int]]:
                List[int]: List of indexes of the first partition.
                List[int]: List of indexes of the second partition.
        """
        # Set user label
        user_label = data.columns[0]

        # Sort by user id label
        df_sorted: DataFrame = data.sort_values(by=[user_label])
        user_counts = df_sorted[user_label].value_counts()

        # Identify all the user with more than k transaction
        users_with_kplus_interactions = user_counts[user_counts > k].index

        # Define test set indices from users with more than k transactions
        test_idxs = (
            df_sorted[df_sorted[user_label].isin(users_with_kplus_interactions)]
            .groupby(user_label)
            .tail(k)
            .index
        )

        # All indexes that are not in test will be in train
        train_idxs = df_sorted.drop(test_idxs).index

        return train_idxs, test_idxs


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
    ) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the temporal holdout splitting. Original data will be splitted
        according to timestamp. If a seed has been set, the split will be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_ratio (float): The test set size.
            val_ratio (Optional[float]): The validation set size.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the test set.
                List[int]: List of indexes that will end up in the validation set.
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
    ) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the temporal leave k out splitting. Original data will be splitted
        according to timestamp. If a seed has been set, the split will be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_k (int): The test set k value.
            val_k (Optional[int]): The validation set k value.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the test set.
                List[int]: List of indexes that will end up in the validation set.
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


@splitting_registry.register(SplittingStrategies.FIXED_TIMESTAMP)
class FixedTimestampSplit(SplittingStrategy):
    """Splits data based on a fixed timestamp.

    Timestamp must be provided to use this strategy.
    """

    def split(
        self,
        data: DataFrame,
        timestamp: int = 0,
        **kwargs: Any,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the fixed timestamp splitting.

        Args:
            data (DataFrame): The DataFrame to be split.
            timestamp (int): The timestamp to split data for test set.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes for the training set.
                List[int]: List of indexes for the test set.
                List[int]: List of indexes for the validation set.
        """
        # Set time label
        time_label = data.columns[-1]  # Assuming last column is timestamp

        # Create a boolean mask for the split
        split_mask = data[time_label] < timestamp  # True for train, False for test

        # Splitting
        train_idxs = data.index[split_mask].tolist()
        test_idxs = data.index[~split_mask].tolist()

        # Return train/test indices
        return train_idxs, test_idxs, None
