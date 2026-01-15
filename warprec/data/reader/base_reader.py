import posixpath
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Dict
from pathlib import Path
from io import StringIO

import pandas as pd
from pandas import DataFrame

from warprec.utils.config import WarpRecConfiguration
from warprec.utils.enums import ReadingMethods
from warprec.utils.logger import logger


class Reader(ABC):
    """The abstract definition of a reader. All readers must extend this class."""

    def _process_tabular_stream(
        self,
        stream: StringIO,
        sep: str,
        header: bool,
        desired_cols: Optional[List[str]] = None,
        desired_dtypes: Optional[Dict[str, str]] = None,
    ) -> DataFrame:
        """Processes tabular data from an in-memory text stream robustly, handling variations
        in columns and data types. This is the shared logic moved from child classes.

        Args:
            stream (StringIO): The in-memory text stream (StringIO) containing the tabular data.
            sep (str): The delimiter character.
            header (bool): A boolean indicating if the data file has a header row.
            desired_cols (Optional[List[str]]): An optional list of column names to select and return.
            desired_dtypes (Optional[Dict[str, str]]): A dictionary mapping column names to desired data types.

        Returns:
            DataFrame: A Pandas DataFrame containing the processed data, adhering to
            `desired_cols` and `desired_dtypes` where possible.
        """
        # pylint: disable = too-many-return-statements, unused-argument
        if desired_dtypes is None:
            desired_dtypes = {}
        # Case 1: The tabular file has a header row
        if header:
            try:
                # First, read only the header to get the column names.
                file_cols = pd.read_csv(stream, sep=sep, nrows=0).columns.tolist()
                stream.seek(0)  # Reset stream for the actual read
            except pd.errors.EmptyDataError:
                logger.attention(
                    "The data stream is empty. Returning an empty DataFrame."
                )
                cols = desired_cols if desired_cols is not None else []
                return pd.DataFrame(columns=cols).astype(desired_dtypes)

            if desired_cols is None:
                return pd.read_csv(stream, sep=sep, header=0, dtype=desired_dtypes)

            cols_to_use = [col for col in desired_cols if col in file_cols]
            if not cols_to_use:
                logger.attention(
                    "None of the desired columns were found. Returning an empty DataFrame."
                )
                return pd.DataFrame()

            dtype_to_use = {
                col: desired_dtypes[col] for col in cols_to_use if col in desired_dtypes
            }
            data = pd.read_csv(
                stream,
                sep=sep,
                header=0,
                usecols=cols_to_use,
                dtype=dtype_to_use if dtype_to_use else None,
            )

        # Case 2: The tabular file does not have a header row
        else:
            try:
                data = pd.read_csv(stream, sep=sep, header=None)
                if data.empty:
                    logger.attention(
                        "The data stream is empty. Returning an empty DataFrame."
                    )
                    cols = desired_cols if desired_cols is not None else []
                    return pd.DataFrame(columns=cols).astype(desired_dtypes)
            except pd.errors.EmptyDataError:
                logger.attention(
                    "The data stream is empty. Returning an empty DataFrame."
                )
                cols = desired_cols if desired_cols is not None else []
                return pd.DataFrame(columns=cols).astype(desired_dtypes)

            if desired_cols is None:
                return data

            num_cols_to_rename = min(len(data.columns), len(desired_cols))
            cols_to_use = desired_cols[:num_cols_to_rename]
            col_mapping = dict(zip(data.columns[:num_cols_to_rename], cols_to_use))
            data.rename(columns=col_mapping, inplace=True)
            data = data[cols_to_use]

            dtype_to_use = {
                col: desired_dtypes[col] for col in cols_to_use if col in desired_dtypes
            }
            if dtype_to_use:
                data = data.astype(dtype_to_use)

        # Final step for both cases
        if desired_cols is not None:
            # Return only requested columns if found
            cols_to_return = [col for col in desired_cols if col in data.columns]
            return data[cols_to_return]

        return data

    @abstractmethod
    def read_tabular(self, *args: Any, **kwargs: Any) -> DataFrame:
        """This method will read the tabular data from the source."""

    @abstractmethod
    def read_tabular_split(self, *args: Any, **kwargs: Any) -> DataFrame:
        """This method will read the tabular split data from the source."""

    # pylint: disable = unused-argument
    def _process_tabular_split(
        self,
        base_location: str,
        column_names: Optional[List[str]] = None,
        dtypes: Optional[Dict[str, str]] = None,
        sep: str = "\t",
        ext: str = ".tsv",
        header: bool = True,
        is_remote: bool = False,  # Flag to handle path joining
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[
        DataFrame, Optional[List[Tuple[DataFrame, DataFrame]] | DataFrame], DataFrame
    ]:
        """This method will read the tabular split data from the source.
        The algorithm is defined here and calls the abstract `read_tabular` method.

        It attempts to load:
        1. Main split: train, validation, and test files from the base location.
        2. K-Fold split: train and validation files from numbered subdirectories (1, 2, ...).

        Args:
            base_location (str): The base path or URI where the split files are located.
            column_names (Optional[List[str]]): A list of expected column names.
            dtypes (Optional[Dict[str, str]]): A dict of data types corresponding to `column_names`.
            sep (str): The delimiter character used in the files. Defaults to tab `\t`.
            ext (str): The file extension. Defaults to `.tsv`.
            header (bool): A boolean indicating if the files have a header row. Defaults to `True`.
            is_remote (bool): If `True`, uses POSIX path joining (for remote URIs/blobs).
                Otherwise, uses `pathlib.Path` for local paths. Defaults to `False`.
            *args (Any): The additional arguments.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            Tuple[DataFrame, Optional[List[Tuple[DataFrame, DataFrame]] | DataFrame], DataFrame]:
                - The main training DataFrame (`train_data`).
                - The main validation DataFrame OR a list of K-fold (train, validation) tuples
                OR `None` if no validation data is found.
                - The main testing DataFrame (`test_data`).

        Raises:
            FileNotFoundError: If the main train or test files are not found or are empty.
        """
        if dtypes is None:
            dtypes = {}

        # Use posixpath for blobs and Path for local files
        path_joiner = posixpath.join if is_remote else lambda *a: str(Path(*a))

        path_main_train = path_joiner(base_location, f"train{ext}")
        path_main_val = path_joiner(base_location, f"validation{ext}")
        path_main_test = path_joiner(base_location, f"test{ext}")

        logger.msg(f"Starting reading split process from: {base_location}")

        train_data = self.read_tabular(
            path_main_train, column_names, dtypes, sep, header
        )
        test_data = self.read_tabular(path_main_test, column_names, dtypes, sep, header)

        if train_data.empty or test_data.empty:
            raise FileNotFoundError(
                f"Train/Test data not found or empty in '{base_location}'."
            )

        # Check for a main validation file first
        # We need a way to check existence without reading the whole file.
        # Let's assume read_tabular returns an empty DataFrame if not found.
        val_data = self.read_tabular(path_main_val, column_names, dtypes, sep, header)
        if not val_data.empty:
            return (train_data, val_data, test_data)

        # Iterate through fold subdirectories
        fold_data = []
        fold_number = 1
        while True:
            fold_path = path_joiner(base_location, str(fold_number))
            path_fold_train = path_joiner(fold_path, f"train{ext}")

            # Try to read the first file of the fold to see if the fold exists
            fold_train = self.read_tabular(
                path_fold_train, column_names, dtypes, sep, header
            )
            if fold_train.empty:
                break  # Stop if a fold is not found

            path_fold_val = path_joiner(fold_path, f"validation{ext}")
            fold_val = self.read_tabular(
                path_fold_val, column_names, dtypes, sep, header
            )

            if not fold_val.empty:
                fold_data.append((fold_train, fold_val))
                fold_number += 1
            else:
                break

        logger.positive("Reading process completed successfully.")
        return (train_data, fold_data if fold_data else None, test_data)

    @abstractmethod
    def read_json(self, *args: Any, **kwargs: Any) -> DataFrame:
        """This method will read the json data from the source."""

    @abstractmethod
    def read_json_split(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[DataFrame, DataFrame | None, DataFrame | None]:
        """This method will read the json split data from the source."""

    @abstractmethod
    def load_model_state(self, *args: Any, **kwargs: Any) -> dict:
        """This method will load a model state from a source."""


class ReaderFactory:  # pylint: disable=C0415, R0903
    """Factory class for creating Reader instances based on configuration."""

    @classmethod
    def get_reader(cls, config: WarpRecConfiguration) -> Reader:
        """Factory method to get the appropriate Reader instance based on the configuration.

        Args:
            config (WarpRecConfiguration): Configuration file.

        Returns:
            Reader: An instance of a class that extends the Reader abstract class.

        Raises:
            ValueError: If the reading method specified in the configuration is unknown.
        """
        reader_type = config.reader.reading_method

        # Create the appropriate Reader instance based on the reading method
        match reader_type:
            case ReadingMethods.LOCAL:
                from warprec.data.reader import LocalReader

                return LocalReader()
            case ReadingMethods.AZURE_BLOB:
                from warprec.data.reader import AzureBlobReader

                storage_account_name = config.general.azure.storage_account_name
                container_name = config.general.azure.container_name

                return AzureBlobReader(
                    storage_account_name=storage_account_name,
                    container_name=container_name,
                )

        raise ValueError(f"Unknown reader type: {reader_type}")
