# pylint: disable=too-few-public-methods
from typing import Tuple, List, Any
from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame
from elliotwo.utils.config import Configuration
from elliotwo.utils.enums import SplittingStrategies
from elliotwo.utils.registry import splitting_registry


class AbstractStrategy(ABC):
    """Abstract definition of a splitting strategy.

    Args:
        config (Configuration): The configuration of the experiment.
        **kwargs (Any): The keyword arguments.

    Attributes:
        read_from_config (bool): Flag to check if the reader is reading from the config file.
    """

    read_from_config: bool = False

    def __init__(self, config: Configuration = None, **kwargs: Any):
        if config:
            self.read_from_config = True
            self._seed = config.general.seed
            self._user_label = config.data.labels.user_id_label
            self._item_label = config.data.labels.item_id_label
            self._time_label = config.data.labels.timestamp_label
            self._test_size = config.splitter.ratio[1]
            self._val_size = config.splitter.ratio[2]
            self._test_k = config.splitter.k[0]
            self._val_k = config.splitter.k[1]

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
class RandomSplit(AbstractStrategy):
    """The definition of the random split strategy.

    This splitting will be executed randomly, unless a seed is set.
    """

    def split(
        self,
        data: DataFrame,
        test_size: float = 0.2,
        val_size: float | None = None,
        **kwargs: Any,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the random splitting. Original data will
        be splitted randomly. If a seed has been set, the split will be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_size (float): The size of the test set.
            val_size (float | None): The size of the validation set.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the validation set.
                List[int]: List of indexes that will end up in the test set.
        """
        # Initialize the variables to be used
        _test_size = self._test_size if self.read_from_config else test_size
        _val_size = self._val_size if self.read_from_config else val_size

        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._ratio_split(data, test_size=_test_size)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if _val_size:
            train_idxs, val_idxs = self._ratio_split(
                data.iloc[train_idxs], test_size=_val_size
            )

        # Otherwise return train/test indices
        return train_idxs, val_idxs, test_idxs

    def _ratio_split(
        self, data: DataFrame, test_size: float = 0.2
    ) -> Tuple[List[int], List[int]]:
        """Method used to split a set of data into two partition,
        respecting the ratio given as input.

        The method used to split data is random. If a seed was set,
        then the split will be reproducible.

        Args:
            data (DataFrame): The original data in DataFrame format.
            test_size (float): This value represent the percentage of
                data that will be taken out.

        Returns:
            Tuple[List[int], List[int]]:
                List[int]: List of indexes of the first partition.
                List[int]: List of indexes of the second partition.

        TODO: This method does not ensures that every user is in the training set.
        """
        np.random.seed(self._seed)

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
class LeaveOneOutSplit(AbstractStrategy):
    """The definition of a leave-one-out splitting strategy."""

    def split(
        self,
        data: DataFrame,
        test_k: int = 1,
        val_k: int | None = None,
        user_label: str = "user_id",
        **kwargs: Any,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the leave-one-out splitting. If a seed has
        been set, the split will be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_k (int): The test k.
            val_k (int | None): The validation k.
            user_label (str): The user label in the DataFrame.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the validation set.
                List[int]: List of indexes that will end up in the test set.
        """
        # Initialize the variables to be used
        _test_k = self._test_k if self.read_from_config else test_k
        _val_k = self._val_k if self.read_from_config else val_k
        _user_label = self._user_label if self.read_from_config else user_label

        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._k_split(data, _test_k, _user_label)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if _val_k:
            train_idxs, val_idxs = self._k_split(
                data.iloc[train_idxs], _val_k, _user_label
            )

        # Otherwise return train/test indices
        return train_idxs, val_idxs, test_idxs

    def _k_split(
        self, data: DataFrame, k: int = 1, user_label: str = "user_id"
    ) -> Tuple[List[int], List[int]]:
        """Method to split data in two partitions, using a fixed number.

        This method will take in account some limit examples like
        users with less then k transactions.

        Args:
            data (DataFrame): The original data in DataFrame format.
            k (int): The number of elements to be included in the second partition.
            user_label (str): The user label in the DataFrame.

        Returns:
            Tuple[List[int], List[int]]:
                List[int]: List of indexes of the first partition.
                List[int]: List of indexes of the second partition.
        """
        # Set random seed for reproducibility
        np.random.seed(self._seed)

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


@splitting_registry.register(SplittingStrategies.TEMPORAL)
class TemporalSplit(AbstractStrategy):
    """The definition of a temporal splitting strategy.

    Timestamp must be provided to use this strategy.
    """

    def split(
        self,
        data: DataFrame,
        test_size: float = 0.2,
        val_size: float | None = None,
        user_label: str = "user_id",
        time_label: str = "user_id",
        **kwargs: Any,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the temporal splitting. Original data will be splitted
        according to timestamp. If a seed has been set, the split will be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_size (float): The test set size.
            val_size (float | None): The validation set size.
            user_label (str): The user label in the DataFrame.
            time_label (str): The timestamp label in the DataFrame.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the validation set.
                List[int]: List of indexes that will end up in the test set.
        """
        # Initialize the variables to be used
        _test_size = self._test_size if self.read_from_config else test_size
        _val_size = self._val_size if self.read_from_config else val_size
        _user_label = self._user_label if self.read_from_config else user_label
        _time_label = self._time_label if self.read_from_config else time_label

        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._temp_split(
            data, test_size=_test_size, user_label=_user_label, time_label=_time_label
        )
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if _val_size:
            train_idxs, val_idxs = self._temp_split(
                data.iloc[train_idxs],
                test_size=_val_size,
                user_label=_user_label,
                time_label=_time_label,
            )

        # Otherwise return train/test indices
        return train_idxs, val_idxs, test_idxs

    def _temp_split(
        self,
        data: DataFrame,
        test_size: float = 0.2,
        user_label: str = "user_id",
        time_label: str = "timestamp",
    ) -> Tuple[List[int], List[int]]:
        """Method to split data in two partitions, using a timestamp.

        This method will split data based on time, using as test
        samples the more recent transactions.

        Args:
            data (DataFrame): The original data in DataFrame format.
            test_size (float): Percentage of data that will end up in the second partition.
            user_label (str): The user label in the DataFrame.
            time_label (str): The timestamp label in the DataFrame.

        Returns:
            Tuple[List[int], List[int]]:
                List[int]: List of indexes of the first partition.
                List[int]: List of indexes of the second partition.
        """
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
