from typing import Tuple, List

import numpy as np
from pandas import DataFrame
from elliotwo.data.dataset import Interactions
from elliotwo.utils.config import Configuration
from elliotwo.utils.enums import SplittingStrategies
from elliotwo.data.dataset import TransactionDataset, ContextDataset
from elliotwo.utils.logger import logger


class Splitter:
    """Splitter class will handle the splitting of the data.

    Args:
        config (Configuration): Configuration file.

    TODO: Implement Strategy Pattern.
    """

    def __init__(self, config: Configuration):
        self._config = config

        # Retrieve informations from config
        if self._config.splitter.ratio:
            self._test_size = self._config.splitter.ratio[1]
            if len(self._config.splitter.ratio) == 3:
                self._val_size = self._config.splitter.ratio[2]
            else:
                self._val_size = None
        self._splitting_strategy = self._config.splitter.strategy
        self._random_seed = self._config.general.seed

        # Retrieve main labels for later use
        self._user_id = self._config.data.labels.user_id_label
        self._item_id = self._config.data.labels.item_id_label

    def split_transaction(self, data: DataFrame) -> TransactionDataset:
        """The main method of the class. This method must be called to split the data.

        When called, this method will return the splitting calculated by \
            the splitting method selected in the configuration file.

        This method accepts transaction data, and will return a TransactionDataset object.

        A transaction is defined by at least a user_id, an item_id.

        Args:
            data (DataFrame): The DataFrame to be splitted.

        Returns:
            TransactionDataset: The dataset for the experiment.

        Raises:
            ValueError: If the splitting strategy is not supported.
        """
        logger.msg(
            f"Starting splitting process with {self._splitting_strategy.value} splitting strategy."
        )

        # Check for splitting strategy and call the right splitting method
        if self._splitting_strategy == SplittingStrategies.RANDOM:
            strategy = self._random_split
        elif self._splitting_strategy == SplittingStrategies.LEAVE_ONE_OUT:
            strategy = self._leave_one_out
        elif self._splitting_strategy == SplittingStrategies.TEMPORAL:
            strategy = self._temporal
        else:
            raise ValueError(
                f"Method {self._splitting_strategy} not supported. \
                    Check documentation for supported splitting strategies."
            )

        # Get indexes using chosen strategy
        idxs = strategy(data)

        # Define train/val/test subset of _inter_df taking into account
        # only user and items present in train set.
        _train_set = data.iloc[idxs[0]]
        if self._config.splitter.validation:
            _val_set = data.iloc[idxs[1]]
            _val_set = _val_set[
                _val_set[self._user_id].isin(_train_set[self._user_id])
                & _val_set[self._item_id].isin(_train_set[self._item_id])
            ]
        _test_set = data.iloc[idxs[2]]
        _test_set = _test_set[
            _test_set[self._user_id].isin(_train_set[self._user_id])
            & _test_set[self._item_id].isin(_train_set[self._item_id])
        ]

        # Define dimensions that will lead the experiment
        _nuid = _train_set[self._user_id].nunique()
        _niid = _train_set[self._item_id].nunique()

        # Update mappings inside Dataset structure
        _uid = _train_set[self._user_id].unique()
        _iid = _train_set[self._item_id].unique()

        # Calculate mapping for users and items
        _umap = {user: i for i, user in enumerate(_uid)}
        _imap = {item: i for i, item in enumerate(_iid)}

        # Create splits inside Dataset class for future use
        train_set = Interactions(_train_set, self._config, (_nuid, _niid), _umap, _imap)
        if self._config.splitter.validation:
            val_set = Interactions(_val_set, self._config, (_nuid, _niid), _umap, _imap)
        else:
            val_set = None
        test_set = Interactions(_test_set, self._config, (_nuid, _niid), _umap, _imap)

        # Logging of splitting process
        logger.msg("Splitting process over.")

        # Train set stats
        train_nuid, train_niid = train_set.get_dims()
        train_transactions = train_set.get_transactions()
        logger.stat_msg(
            (
                f"Number of users: {train_nuid}      "
                f"Number of items: {train_niid}      "
                f"Transactions: {train_transactions}"
            ),
            "Train set",
        )
        if self._config.splitter.validation:
            val_nuid, val_niid = val_set.get_dims()
            val_transactions = val_set.get_transactions()
            logger.stat_msg(
                (
                    f"Number of users: {val_nuid}      "
                    f"Number of items: {val_niid}      "
                    f"Transactions: {val_transactions}"
                ),
                "Train set",
            )
        test_nuid, test_niid = test_set.get_dims()
        test_transactions = test_set.get_transactions()
        logger.stat_msg(
            (
                f"Number of users: {test_nuid}      "
                f"Number of items: {test_niid}      "
                f"Transactions: {test_transactions}"
            ),
            "Train set",
        )

        return TransactionDataset(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            config=self._config,
            user_mapping=_umap,
            item_mapping=_imap,
            nuid=_nuid,
            niid=_niid,
        )

    def split_context(self, data: DataFrame) -> ContextDataset:
        raise NotImplementedError

    def _random_split(self, data: DataFrame) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the random splitting. Original data will \
            be splitted randomly. If a seed has benn set, the split wiil be reproducible.

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
        if self._config.splitter.validation:
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
        np.random.seed(self._random_seed)

        user_groups = data.groupby(
            "user_id"
        ).indices  # Dictionary {user_id: np.array(indices)}

        train_indices = []
        test_indices = []

        for user, indices in user_groups.items():
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

    def _leave_one_out(self, data: DataFrame) -> Tuple[List[int], List[int], List[int]]:
        """Implementation of the leave-one-out splitting. If a seed has \
            been set, the split wiil be reproducible.

        Args:
            data (DataFrame): The DataFrame to be splitted.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                List[int]: List of indexes that will end up in the training set.
                List[int]: List of indexes that will end up in the validation set.
                List[int]: List of indexes that will end up in the test set.
        """
        # If k was not set, leave one out strategy will be used
        k = self._config.splitter.k if self._config.splitter.k else 1

        # Get interactions in DataFrame and calculate train/test indices
        train_idxs, test_idxs = self._k_split(data, k)
        val_idxs = None

        # Check if validation set size has been set, if so we return indices for train/val/test
        if self._config.splitter.validation:
            train_idxs, val_idxs = self._k_split(data.iloc[train_idxs], k)

        # Otherwise return train/test indices
        return train_idxs, val_idxs, test_idxs

    def _k_split(self, data: DataFrame, k: int = 1) -> Tuple[List[int], List[int]]:
        """Method to split data in two partitions, using a fixed number.

        This method will take in account some limit examples like \
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
        np.random.seed(self._random_seed)

        # Sort by user id label
        user_id_label = self._config.data.labels.user_id_label
        df_sorted = data.sort_values(by=[user_id_label])
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

    def _temporal(self, data: DataFrame) -> Tuple[List[int], List[int], List[int]]:
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
        if self._config.splitter.validation:
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
        split_indices = np.floor(user_counts * (1 - test_size)).astype(int)

        # Generate a mask to efficiently split data
        split_mask = (
            data.groupby(user_label).cumcount()
            < split_indices.loc[data[user_label]].values
        )

        # Splitting
        train_idxs = data.index[split_mask].tolist()
        test_idxs = data.index[~split_mask].tolist()

        return train_idxs, test_idxs
