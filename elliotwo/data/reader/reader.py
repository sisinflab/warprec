from typing import Any
from os.path import join, isfile
from abc import ABC, abstractmethod

import pandas as pd
import joblib
from pathlib import Path
from pandas import DataFrame
from elliotwo.data.dataset import Interactions, TransactionDataset
from elliotwo.utils.config import Configuration
from elliotwo.utils.logger import logger


class AbstractReader(ABC):
    """The abstract definition of a reader. All readers must extend this class.

    Args:
        config (Configuration): Configuration file.

    TODO: Use Factory Pattern for different reader.
    """

    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.data = None

    @abstractmethod
    def read(self) -> DataFrame:
        """This method will read the data from the source.

        Returns:
            DataFrame: The data read from the source.
        """

    @abstractmethod
    def load_model_state(self, **kwargs) -> dict:
        """This method will load a model state from a source."""

    @abstractmethod
    def read_transaction_split(self) -> TransactionDataset:
        """This method will read the split data from the source.

        Returns:
            TransactionDataset: The dataset object containing the split transaction data.
        """

    def get_raw(self) -> DataFrame:
        """This method will return the raw data.

        Returns:
            DataFrame: The raw data, as it was read from the source.
        """
        return self.data


class LocalReader(AbstractReader):
    """This class extends the AbstractReader, it handles
    all the data reading part from a local machine.

    Args:
        config (Configuration): Configuration file.
    """

    def __init__(self, config: Configuration) -> None:
        super().__init__(config)

        # Retrieve the path from the config. This isn't an optional value
        self._path = config.data.local_path

        # Check if the optional reading parameters have been set
        self._sep = config.data.sep
        self._batch_size = config.data.batch_size
        self._column_names = config.column_names()
        self._dtypes = dict(zip(self._column_names, config.column_dtype()))

        # Check if the optional label parameters have been set
        self._user_label = config.data.labels.user_id_label
        self._item_label = config.data.labels.item_id_label
        self._score_label = config.data.labels.rating_label
        self._timestamp_label = config.data.labels.timestamp_label

    def read(self) -> DataFrame:
        """This method will read the data locally, using parameters from the config file.

        Returns:
            DataFrame: The data read from the local source.
        """
        logger.msg(f"Starting reading process from local source in: {self._path}")
        data = None

        if self._batch_size is not None:
            # In case batch_size has been set, read data in batches and then create the dataframe
            chunks = []
            for chunk in pd.read_csv(
                self._path,
                sep=self._sep,
                chunksize=self._batch_size,
                usecols=self._column_names,
                dtype=self._dtypes,
            ):
                chunks.append(chunk)
            data = pd.concat(chunks, ignore_index=True)
        else:
            # In case batch_size hasn't been set, read the data in one go
            data = pd.read_csv(
                self._path,
                sep=self._sep,
                usecols=self._column_names,
                dtype=self._dtypes,
            )
        logger.msg("Data loaded correctly from local source.")

        return data

    def load_model_state(self, **kwargs: Any) -> dict:
        """This method will load a model state from a given path.

        Args:
            **kwargs (Any): The keyword arguments. For local reading 'local_path' must be provided.

        Returns:
            dict: The deserialized information of the model.

        Raises:
            ValueError: If the path was not provided.
            FileNotFoundError: If the model state was not found in the provided path.
        """
        if "local_path" not in kwargs:
            raise ValueError("Local path to model state was not provided.")
        path = Path(kwargs["local_path"])
        if path.exists():
            return joblib.load(path)
        raise FileNotFoundError(f"Model state not found in {path}")

    def read_transaction_split(self) -> TransactionDataset:
        """This method will read the split data from the local source,
        using parameters from the config file.

        Returns:
            TransactionDataset: The dataset object containing the split transaction data.

        Raises:
            FileNotFoundError: If the train split file was not found.
        """
        _path_train = join(self.config.data.split_dir, "split", "train.tsv")
        _path_test = join(self.config.data.split_dir, "split", "test.tsv")
        _path_val = join(self.config.data.split_dir, "split", "val.tsv")

        train_inter = None
        test_inter = None
        val_inter = None

        # Get batch size from config
        batch_size = self.config.data.batch_size

        logger.msg(f"Starting reading process from local source in: {_path_train}")
        if isfile(_path_train):
            train_set = pd.read_csv(
                _path_train,
                sep=self._sep,
                usecols=self._column_names,
                dtype=self._dtypes,
            )
            _nuid = train_set[self._user_label].nunique()
            _niid = train_set[self._item_label].nunique()
            _umap = {
                user: i for i, user in enumerate(train_set[self._user_label].unique())
            }
            _imap = {
                item: i for i, item in enumerate(train_set[self._item_label].unique())
            }
            train_inter = Interactions(
                train_set,
                self.config,
                (_nuid, _niid),
                _umap,
                _imap,
                batch_size=batch_size,
            )

            if isfile(_path_test):
                test_set = pd.read_csv(
                    _path_test,
                    sep=self._sep,
                    usecols=self._column_names,
                    dtype=self._dtypes,
                )

                test_inter = Interactions(
                    test_set,
                    self.config,
                    (_nuid, _niid),
                    _umap,
                    _imap,
                    batch_size=batch_size,
                )

            if isfile(_path_val):
                val_set = pd.read_csv(
                    _path_val,
                    sep=self._sep,
                    usecols=self._column_names,
                    dtype=self._dtypes,
                )

                val_inter = Interactions(
                    val_set,
                    self.config,
                    (_nuid, _niid),
                    _umap,
                    _imap,
                    batch_size=batch_size,
                )

            train_nuid, train_niid = train_inter.get_dims()
            train_transactions = train_inter.get_transactions()
            logger.stat_msg(
                (
                    f"Number of users: {train_nuid}      "
                    f"Number of items: {train_niid}      "
                    f"Transactions: {train_transactions}"
                ),
                "Train split information",
            )
            if test_inter is not None:
                test_nuid, test_niid = test_inter.get_dims()
                test_transactions = test_inter.get_transactions()
                logger.stat_msg(
                    (
                        f"Number of users: {test_nuid}      "
                        f"Number of items: {test_niid}      "
                        f"Transactions: {test_transactions}"
                    ),
                    "Test split information",
                )
            if val_inter is not None:
                val_nuid, val_niid = val_inter.get_dims()
                val_transactions = val_inter.get_transactions()
                logger.stat_msg(
                    (
                        f"Number of users: {val_nuid}      "
                        f"Number of items: {val_niid}      "
                        f"Transactions: {val_transactions}"
                    ),
                    "Validation split information",
                )

            return TransactionDataset(
                train_inter,
                _umap,
                _imap,
                _nuid,
                _niid,
                self.config,
                val_set=val_inter,
                test_set=test_inter,
            )
        raise FileNotFoundError(f"Train split not found in {_path_train}")
