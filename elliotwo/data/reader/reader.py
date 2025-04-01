from typing import Any, List
from os.path import join, isfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import joblib
from pandas import DataFrame
from elliotwo.data.dataset import Interactions, TransactionDataset
from elliotwo.utils.config import (
    Configuration,
    ReaderConfig,
    Labels,
    CustomDtype,
    ReadingParams,
    SplitReading,
)
from elliotwo.utils.enums import RatingType, ReadingMethods
from elliotwo.utils.logger import logger


class AbstractReader(ABC):
    """The abstract definition of a reader. All readers must extend this class.

    Attributes:
        config (Configuration): Configuration file.

    TODO: Use Factory Pattern for different reader.
    """

    config: Configuration = None

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
            self.config = config

    def read(
        self,
        local_path: str = None,
        rating_type: RatingType = RatingType.IMPLICIT,
        sep: str = "\t",
        batch_size: int = 1024,
        column_names: List[str] | None = None,
        dtypes: List[str] | None = None,
        **kwargs: Any,
    ) -> DataFrame:
        """This method will read the data locally, using parameters from the config file.

        Args:
            local_path (str): The path to the local file.
            rating_type (RatingType): The rating type used in the dataset.
            sep (str): The separator used in the file.
            batch_size (int): The batch size that will be used to
                iterate over the interactions.
            column_names (List[str] | None): The column names of the data.
            dtypes (List[str] | None): The data types of the columns.
            **kwargs (Any): The keyword arguments.

        Returns:
            DataFrame: The data read from the local source.
        """
        if self.config:
            read_config = self.config.reader
        else:
            if not column_names:
                column_names = ["user_id", "item_id", "rating", "timestamp"]
            if not dtypes:
                dtypes = ["int32", "int32", "float32", "int32"]
            dtypes_map = {name: dtype for name, dtype in zip(column_names, dtypes)}

            read_config = ReaderConfig(
                loading_strategy="dataset",
                data_type="transaction",
                reading_method=ReadingMethods.LOCAL,
                local_path=local_path,
                rating_type=rating_type,
                labels=Labels.from_list(column_names),
                dtypes=CustomDtype(**dtypes_map),
                reading_params=ReadingParams(sep=sep, batch_size=batch_size),
            )

        logger.msg(
            f"Starting reading process from local source in: {read_config.local_path}"
        )

        if read_config.reading_params.batch_size is not None:
            # In case batch_size has been set, read data in batches and then create the dataframe
            chunks = []
            for chunk in pd.read_csv(
                read_config.local_path,
                sep=read_config.reading_params.sep,
                chunksize=read_config.reading_params.batch_size,
                usecols=read_config.column_names(),
                dtype=read_config.column_dtype(),
            ):
                chunks.append(chunk)
            data = pd.concat(chunks, ignore_index=True)
        else:
            # In case batch_size hasn't been set, read the data in one go
            data = pd.read_csv(
                read_config.local_path,
                sep=read_config.reading_params.sep,
                usecols=read_config.column_names(),
                dtype=read_config.column_dtype(),
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
        if self.config:
            read_config = self.config.reader
        else:
            read_config = ReaderConfig(
                loading_strategy="dataset",
                data_type="dataset",
                reading_method=ReadingMethods.LOCAL,
                rating_type=rating_type,
                labels=Labels(*column_names),
                dtypes=CustomDtype(*dtypes),
                split=SplitReading(
                    local_path=split_dir, ext=ext, sep=sep, batch_size=batch_size
                ),
            )

        # Define the paths to the split files
        path_train = join(read_config.split.local_path, "train" + read_config.split.ext)
        path_test = join(read_config.split.local_path, "test" + read_config.split.ext)
        path_val = join(read_config.split.local_path, "val" + read_config.split.ext)

        train_inter = None
        test_inter = None
        val_inter = None

        logger.msg(f"Starting reading process from local source in: {path_train}")
        if isfile(path_train):
            train_set = pd.read_csv(
                path_train,
                sep=read_config.split.sep,
                usecols=read_config.column_names(),
                dtype=read_config.column_dtype(),
            )
            _nuid = train_set[read_config.labels.user_id_label].nunique()
            _niid = train_set[read_config.labels.item_id_label].nunique()
            _umap = {
                user: i
                for i, user in enumerate(
                    train_set[read_config.labels.user_id_label].unique()
                )
            }
            _imap = {
                item: i
                for i, item in enumerate(
                    train_set[read_config.labels.item_id_label].unique()
                )
            }
            train_inter = Interactions(
                train_set,
                (_nuid, _niid),
                _umap,
                _imap,
                batch_size=read_config.split.batch_size,
                user_id_label=read_config.labels.user_id_label,
                item_id_label=read_config.labels.item_id_label,
                rating_label=read_config.labels.rating_label,
                rating_type=read_config.rating_type,
            )

            if isfile(path_test):
                test_set = pd.read_csv(
                    path_test,
                    sep=read_config.split.sep,
                    usecols=read_config.column_names(),
                    dtype=read_config.column_dtype(),
                )

                test_inter = Interactions(
                    test_set,
                    (_nuid, _niid),
                    _umap,
                    _imap,
                    batch_size=read_config.split.batch_size,
                    user_id_label=read_config.labels.user_id_label,
                    item_id_label=read_config.labels.item_id_label,
                    rating_label=read_config.labels.rating_label,
                    rating_type=read_config.rating_type,
                )

            if isfile(path_val):
                val_set = pd.read_csv(
                    path_val,
                    sep=read_config.split.sep,
                    usecols=read_config.column_names(),
                    dtype=read_config.column_dtype(),
                )

                val_inter = Interactions(
                    val_set,
                    (_nuid, _niid),
                    _umap,
                    _imap,
                    batch_size=read_config.split.batch_size,
                    user_id_label=read_config.labels.user_id_label,
                    item_id_label=read_config.labels.item_id_label,
                    rating_label=read_config.labels.rating_label,
                    rating_type=read_config.rating_type,
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
