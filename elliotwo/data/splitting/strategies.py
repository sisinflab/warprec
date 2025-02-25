from typing import Tuple, List
from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame
from elliotwo.utils.config import Configuration
from elliotwo.utils.registry import splitting_registry


class AbstractStrategy(ABC):
    """Abstract definition of a splitting strategy.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def __init__(self, config: Configuration):
        self._config = config
        self._seed = config.general.seed

    @abstractmethod
    def split(self, data: DataFrame) -> Tuple[List[int], List[int], List[int]]:
        """This method will split the data in train/test/validation splits.

        If the validation split was not set then it will be None.

        Args:
            data (DataFrame): The DataFrame to be splitted.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the test set.
                List[int]: List of indexes that will end up in the validation set.
        """


@splitting_registry.register("random")
class RandomSplit(AbstractStrategy):
    """The definition of the random split strategy.

    This splitting will be executed randomly, unless a seed is set.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._test_size = config.splitter.ratio[1]
        self._validation = config.splitter.validation
        if self._validation:
            self._val_size = config.splitter.ratio[2]

    def split(self, data: DataFrame) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the random splitting. Original data will
            be splitted randomly. If a seed has been set, the split wiil be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the validation set.
                List[int]: List of indexes that will end up in the test set.
        """
        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._ratio_split(data, test_size=self._test_size)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if self._validation:
            train_idxs, val_idxs = self._ratio_split(
                data.iloc[train_idxs], test_size=self._val_size
            )

        # Otherwise return train/test indices
        return train_idxs, val_idxs, test_idxs

    def _ratio_split(
        self, data: DataFrame, test_size: float = 0.2
    ) -> Tuple[List[int], List[int]]:
        """Method used to split a set of data into two partition,
            respecting the rateo given as input.

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


@splitting_registry.register("leave-one-out")
class LeaveOneOutSplit(AbstractStrategy):
    """The definition of a leave-one-out splitting strategy.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._k = config.splitter.k
        self._validation = config.splitter.validation

    def split(self, data: DataFrame) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the leave-one-out splitting. If a seed has
            been set, the split wiil be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the validation set.
                List[int]: List of indexes that will end up in the test set.
        """
        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._k_split(data, self._k)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if self._validation:
            train_idxs, val_idxs = self._k_split(data.iloc[train_idxs], self._k)

        # Otherwise return train/test indices
        return train_idxs, val_idxs, test_idxs

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
        # Set random seed for reproducibility
        np.random.seed(self._seed)

        # Sort by user id label
        user_id_label = self._config.data.labels.user_id_label
        df_sorted: DataFrame = data.sort_values(by=[user_id_label])
        user_counts = df_sorted[user_id_label].value_counts()

        # Identify all the user with more than k transaction
        users_with_kplus_interactions = user_counts[user_counts > k].index

        # Define test set indices from users with more than k transactions
        test_idxs = (
            df_sorted[df_sorted[user_id_label].isin(users_with_kplus_interactions)]
            .groupby(user_id_label)
            .tail(k)
            .index
        )

        # All indexes that are not in test will be in train
        train_idxs = df_sorted.drop(test_idxs).index

        return train_idxs, test_idxs


@splitting_registry.register("temporal")
class TemporalSplit(AbstractStrategy):
    """The definition of a temporal splitting strategy.

    Timestamp must be provided to use this strategy.

    Args:
        config (Configuration): The configuration of the experiment.
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._test_size = config.splitter.ratio[1]
        self._validation = config.splitter.validation
        if self._validation:
            self._val_size = config.splitter.ratio[2]

    def split(self, data: DataFrame) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the temporal splitting. Original data will be splitted \
            according to timestamp. If a seed has benn set, the split wiil be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the validation set.
                List[int]: List of indexes that will end up in the test set.
        """
        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._temp_split(data, test_size=self._test_size)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if self._validation:
            train_idxs, val_idxs = self._temp_split(
                data.iloc[train_idxs], test_size=self._val_size
            )

        # Otherwise return train/test indices
        return train_idxs, val_idxs, test_idxs

    def _temp_split(
        self, data: DataFrame, test_size: float = 0.2
    ) -> Tuple[List[int], List[int]]:
        """Method to split data in two partitions, using a timestamp.

        This method will split data based on time, using as test\
            samples the more recent transactions.

        Args:
            data (DataFrame): The original data in DataFrame format.
            test_size (float): Percentage of data that will end up in the second partition.

        Returns:
            Tuple[List[int], List[int]]:
                List[int]: List of indexes of the first partition.
                List[int]: List of indexes of the second partition.
        """
        user_label = self._config.data.labels.user_id_label
        time_label = self._config.data.labels.timestamp_label

        # Single sorting by user and timestamp
        data = data.sort_values(by=[user_label, time_label])

        # Calculate index where to split
        user_counts = data[user_label].value_counts().sort_index()
        split_indices: DataFrame = np.floor(user_counts * (1 - test_size)).astype(int)

        # Generate a mask to efficiently split data
        split_mask = (
            data.groupby(user_label).cumcount()
            < split_indices.loc[data[user_label]].values
        )

        # Splitting
        train_idxs = data.index[split_mask].tolist()
        test_idxs = data.index[~split_mask].tolist()

        return train_idxs, test_idxs
