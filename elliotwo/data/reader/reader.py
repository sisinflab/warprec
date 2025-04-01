from typing import Any, List
from os.path import join, isfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
from pandas import DataFrame
from elliotwo.data.dataset import Interactions, TransactionDataset
from elliotwo.utils.config import Configuration
from elliotwo.utils.enums import RatingType
from elliotwo.utils.logger import logger


class AbstractReader(ABC):
    """The abstract definition of a reader. All readers must extend this class.

    Args:
        config (Configuration): Configuration file.

    Attributes:
        read_from_config (bool): Flag to check if the reader is reading from the config file.

    TODO: Use Factory Pattern for different reader.
    """

    read_from_config: bool = False

    @abstractmethod
    def read(self, **kwargs: Any) -> DataFrame:
        """This method will read the data from the source."""

    @abstractmethod
    def load_model_state(self, **kwargs: Any) -> dict:
        """This method will load a model state from a source."""

    @abstractmethod
    def read_transaction_split(self, **kwargs: Any) -> TransactionDataset:
        """This method will read the split data from the source."""


class LocalReader(AbstractReader):
    """This class extends the AbstractReader, it handles
    all the data reading part from a local machine.

    Args:
        config (Configuration): Configuration file.
    """

    def __init__(self, config: Configuration = None) -> None:
        if config:
            self.read_from_config = True

            # Retrieve the path from the config. This isn't an optional value
            self.path = config.reader.local_path

            # Check if the optional reading parameters have been set
            self.sep = config.reader.reading_params.sep
            self.batch_size = config.reader.reading_params.batch_size
            self.column_names = config.column_names()
            self.dtypes = dict(zip(self.column_names, config.column_dtype()))
            self.rating_type = config.reader.rating_type

            # Split parameters
            self.split_dir = config.reader.split.local_path
            self.split_ext = config.reader.split.ext
            self.split_sep = config.reader.split.sep

            # Check if the optional label parameters have been set
            self.user_label = config.reader.labels.user_id_label
            self.item_label = config.reader.labels.item_id_label
            self.rating_label = config.reader.labels.rating_label

    def read(
        self,
        local_path: str = None,
        sep: str = ",",
        batch_size: int = 1024,
        column_names: List[str] | None = None,
        dtypes: List[str] | None = None,
        **kwargs: Any,
    ) -> DataFrame:
        """This method will read the data locally, using parameters from the config file.

        Args:
            local_path (str): The path to the local file.
            sep (str): The separator used in the file.
            batch_size (int): The batch size that will be used to
                iterate over the interactions.
            column_names (List[str] | None): The column names of the data.
            dtypes (List[str] | None): The data types of the columns.
            **kwargs (Any): The keyword arguments.

        Returns:
            DataFrame: The data read from the local source.
        """
        # Default values
        column_names = (
            column_names
            if column_names is not None
            else ["user_id", "item_id", "rating", "timestamp"]
        )
        dtypes = (
            dtypes if dtypes is not None else ["int32", "int32", "float32", "int32"]
        )

        # Initialize the variables to be used
        _path = self.path if self.read_from_config else local_path
        _sep = self.sep if self.read_from_config else sep
        _batch_size = self.batch_size if self.read_from_config else batch_size
        _column_names = self.column_names if self.read_from_config else column_names
        _dtypes = (
            self.dtypes
            if self.read_from_config
            else dict(zip(_column_names, map(np.dtype, dtypes)))
        )

        logger.msg(f"Starting reading process from local source in: {_path}")

        if _batch_size is not None:
            # In case batch_size has been set, read data in batches and then create the dataframe
            chunks = []
            for chunk in pd.read_csv(
                _path,
                sep=_sep,
                chunksize=_batch_size,
                usecols=_column_names,
                dtype=_dtypes,
            ):
                chunks.append(chunk)
            data = pd.concat(chunks, ignore_index=True)
        else:
            # In case batch_size hasn't been set, read the data in one go
            data = pd.read_csv(
                _path,
                sep=_sep,
                usecols=_column_names,
                dtype=_dtypes,
            )
        logger.msg("Data loaded correctly from local source.")

        return data

    def load_model_state(self, local_path: str = None, **kwargs: Any) -> dict:
        """This method will load a model state from a given path.

        Args:
            local_path (str): The path to the model state.
            **kwargs (Any): The keyword arguments.

        Returns:
            dict: The deserialized information of the model.

        Raises:
            FileNotFoundError: If the model state was not found in the provided path.
            ValueError: If the path to the model state has not been provided.
        """
        if local_path:
            path = Path(local_path)
            if path.exists():
                return joblib.load(path)
            raise FileNotFoundError(f"Model state not found in {path}")
        raise ValueError("Local path to model state must be provided.")

    def read_transaction_split(
        self,
        split_dir: str = None,
        sep: str = ",",
        ext: str = ".csv",
        batch_size: int = 1024,
        column_names: List[str] | None = None,
        dtypes: List[str] | None = None,
        rating_type: RatingType = RatingType.IMPLICIT,
        **kwargs: Any,
    ) -> TransactionDataset:
        """This method will read the split data from the local source,
        using parameters from the config file.

        Args:
            split_dir (str): The path to the split directory.
            sep (str): The separator used in the file.
            ext (str): The extension of the split files.
            batch_size (int): The batch size that will be used to
                iterate over the interactions.
            column_names (List[str] | None): The column names of the data.
            dtypes (List[str] | None): The data types of the columns.
            rating_type (RatingType): The type of rating to be used.
            **kwargs (Any): The keyword arguments.

        Returns:
            TransactionDataset: The dataset object containing the split transaction data.

        Raises:
            FileNotFoundError: If the train split file was not found.
        """
        # Default values
        column_names = (
            column_names
            if column_names is not None
            else ["user_id", "item_id", "rating", "timestamp"]
        )
        dtypes = (
            dtypes if dtypes is not None else ["int32", "int32", "float32", "int32"]
        )

        # Initialize the variables to be used
        _split_dir = self.split_dir if self.read_from_config else split_dir
        _sep = self.split_sep if self.read_from_config else sep
        _ext = self.split_ext if self.read_from_config else ext
        _batch_size = self.batch_size if self.read_from_config else batch_size
        _column_names = self.column_names if self.read_from_config else column_names
        _dtypes = (
            self.dtypes
            if self.read_from_config
            else dict(zip(_column_names, map(np.dtype, dtypes)))
        )
        _user_label = self.user_label if self.read_from_config else _column_names[0]
        _item_label = self.item_label if self.read_from_config else _column_names[1]
        _rating_type = self.rating_type if self.read_from_config else rating_type
        if _rating_type == RatingType.EXPLICIT:
            _rating_label = (
                self.rating_label if self.read_from_config else _column_names[2]
            )
        else:
            _rating_label = None

        # Define the paths to the split files
        path_train = join(_split_dir, "train" + _ext)
        path_test = join(_split_dir, "test" + _ext)
        path_val = join(_split_dir, "val" + _ext)

        train_inter = None
        test_inter = None
        val_inter = None

        logger.msg(f"Starting reading process from local source in: {path_train}")
        if isfile(path_train):
            train_set = pd.read_csv(
                path_train,
                sep=_sep,
                usecols=_column_names,
                dtype=_dtypes,
            )
            _nuid = train_set[_user_label].nunique()
            _niid = train_set[_item_label].nunique()
            _umap = {user: i for i, user in enumerate(train_set[_user_label].unique())}
            _imap = {item: i for i, item in enumerate(train_set[_item_label].unique())}
            train_inter = Interactions(
                train_set,
                (_nuid, _niid),
                _umap,
                _imap,
                batch_size=_batch_size,
                user_id_label=_user_label,
                item_id_label=_item_label,
                rating_label=_rating_label,
                rating_type=_rating_type,
            )

            if isfile(path_test):
                test_set = pd.read_csv(
                    path_test,
                    sep=_sep,
                    usecols=_column_names,
                    dtype=_dtypes,
                )

                test_inter = Interactions(
                    test_set,
                    (_nuid, _niid),
                    _umap,
                    _imap,
                    batch_size=_batch_size,
                    user_id_label=_user_label,
                    item_id_label=_item_label,
                    rating_label=_rating_label,
                    rating_type=_rating_type,
                )

            if isfile(path_val):
                val_set = pd.read_csv(
                    path_val,
                    sep=_sep,
                    usecols=_column_names,
                    dtype=_dtypes,
                )

                val_inter = Interactions(
                    val_set,
                    (_nuid, _niid),
                    _umap,
                    _imap,
                    batch_size=_batch_size,
                    user_id_label=_user_label,
                    item_id_label=_item_label,
                    rating_label=_rating_label,
                    rating_type=_rating_type,
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
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                user_mapping=_umap,
                item_mapping=_imap,
                nuid=_nuid,
                niid=_niid,
            )
        raise FileNotFoundError(f"Train split not found in {path_train}")
