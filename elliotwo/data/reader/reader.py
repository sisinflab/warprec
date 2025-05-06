from typing import Any, List, Tuple
from os.path import join, isfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import joblib
from pandas import DataFrame
from elliotwo.utils.config import (
    Configuration,
    ReaderConfig,
    Labels,
    CustomDtype,
    SplitReading,
    SideInformationReading,
)
from elliotwo.utils.enums import RatingType, ReadingMethods
from elliotwo.utils.logger import logger


class Reader(ABC):
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
    def read_transaction_split(
        self, **kwargs: Any
    ) -> Tuple[DataFrame, DataFrame | None, DataFrame | None]:
        """This method will read the split data from the source."""


class LocalReader(Reader):
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
        column_names: List[str] | None = None,
        dtypes: List[str] | None = None,
        **kwargs: Any,
    ) -> DataFrame:
        """This method will read the data locally, using parameters from the config file.

        Args:
            local_path (str): The path to the local file.
            rating_type (RatingType): The rating type used in the dataset.
            sep (str): The separator used in the file.
            column_names (List[str] | None): The column names of the data.
            dtypes (List[str] | None): The data types of the columns.
            **kwargs (Any): The keyword arguments.

        Returns:
            DataFrame: The data read from the local source.
        """
        if self.config:
            read_config = self.config.reader
        else:
            # Column names and dtypes default values are here for mypy waring
            if not column_names:
                column_names = ["user_id", "item_id", "rating", "timestamp"]
            if not dtypes:
                dtypes = ["int32", "int32", "float32", "int32"]
            dtypes_map = dict(zip(column_names, dtypes))

            read_config = ReaderConfig(
                loading_strategy="dataset",
                data_type="transaction",
                reading_method=ReadingMethods.LOCAL,
                local_path=local_path,
                sep=sep,
                rating_type=rating_type,
                labels=Labels.from_list(column_names),
                dtypes=CustomDtype(**dtypes_map),
            )

        logger.msg(
            f"Starting reading process from local source in: {read_config.local_path}"
        )
        data = pd.read_csv(
            read_config.local_path,
            sep=read_config.sep,
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
        sep: str = "\t",
        ext: str = ".tsv",
        column_names: List[str] | None = None,
        dtypes: List[str] | None = None,
        rating_type: RatingType = RatingType.IMPLICIT,
        **kwargs: Any,
    ) -> Tuple[DataFrame, DataFrame | None, DataFrame | None]:
        """This method will read the split data from the local source,
        using parameters from the config file.

        Args:
            split_dir (str): The path to the split directory.
            sep (str): The separator used in the file.
            ext (str): The extension of the split files.
            column_names (List[str] | None): The column names of the data.
            dtypes (List[str] | None): The data types of the columns.
            rating_type (RatingType): The type of rating to be used.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[DataFrame, DataFrame | None, DataFrame | None]:
                DataFrame: The training set data.
                DataFrame: The test set data if present.
                DataFrame | None: The validation set data if present.

        Raises:
            FileNotFoundError: If the train split file was not found.
        """
        if self.config:
            read_config = self.config.reader
        else:
            # Column names and dtypes default values are here for mypy waring
            if not column_names:
                column_names = ["user_id", "item_id", "rating", "timestamp"]
            if not dtypes:
                dtypes = ["int32", "int32", "float32", "int32"]
            dtypes_map = dict(zip(column_names, dtypes))

            read_config = ReaderConfig(
                loading_strategy="split",
                data_type="transaction",
                reading_method=ReadingMethods.LOCAL,
                rating_type=rating_type,
                labels=Labels.from_list(column_names),
                dtypes=CustomDtype(**dtypes_map),
                split=SplitReading(local_path=split_dir, ext=ext, sep=sep),
            )

        # Define the paths to the split files
        path_train = join(read_config.split.local_path, "train" + read_config.split.ext)
        path_test = join(read_config.split.local_path, "test" + read_config.split.ext)
        path_val = join(read_config.split.local_path, "val" + read_config.split.ext)

        logger.msg(f"Starting reading process from local source in: {path_train}")

        # DataFrame init
        train_data = None
        test_data = None
        val_data = None

        if isfile(path_train):
            train_data = pd.read_csv(
                path_train,
                sep=read_config.split.sep,
                usecols=read_config.column_names(),
                dtype=read_config.column_dtype(),
            )

            if isfile(path_test):
                test_data = pd.read_csv(
                    path_test,
                    sep=read_config.split.sep,
                    usecols=read_config.column_names(),
                    dtype=read_config.column_dtype(),
                )

            if isfile(path_val):
                val_data = pd.read_csv(
                    path_val,
                    sep=read_config.split.sep,
                    usecols=read_config.column_names(),
                    dtype=read_config.column_dtype(),
                )

            return (train_data, test_data, val_data)
        raise FileNotFoundError(f"Train split not found in {path_train}")

    def read_side_information(
        self,
        local_path: str = None,
        sep: str = "\t",
        **kwargs: Any,
    ) -> DataFrame:
        """This method will read the side information locally, using parameters from the config file.

        Args:
            local_path (str): The path to the local file.
            sep (str): The separator used in the file.
            **kwargs (Any): The keyword arguments.

        Returns:
            DataFrame: The data read from the local source.
        """
        if self.config:
            read_config = self.config.reader.side
        else:
            read_config = SideInformationReading(local_path=local_path, sep=sep)

        logger.msg(
            f"Reading side information from local source in: {read_config.local_path}"
        )
        data = pd.read_csv(read_config.local_path, sep=read_config.sep)
        logger.msg("Data loaded correctly from local source.")

        return data
