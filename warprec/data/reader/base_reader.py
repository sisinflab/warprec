import posixpath
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Dict, Union
from pathlib import Path
from io import StringIO, BytesIO

import pandas as pd
import polars as pl
import narwhals as nw
from narwhals.dataframe import DataFrame

from warprec.utils.config import WarpRecConfiguration
from warprec.utils.enums import ReadingMethods
from warprec.utils.logger import logger


class Reader(ABC):
    """The abstract definition of a reader. All readers must extend this class.

    Args:
        backend (str): The backend to use for reading data.

    Raises:
        ValueError: If the backend is not supported.
    """

    def __init__(self, backend: str = "polars"):
        self.backend = backend.lower()

        if self.backend not in ["polars", "pandas"]:
            raise ValueError(
                f"Initializing reader module with a not supported backend: {self.backend}."
            )

    def _process_tabular_data(
        self,
        source: Union[str, Path, StringIO, BytesIO],
        sep: str,
        header: bool,
        desired_cols: Optional[List[str]] = None,
        desired_dtypes: Optional[Dict[str, str]] = None,
    ) -> DataFrame[Any]:
        """Processes tabular data from a source (path or stream) based on the selected backend.

        Args:
            source (Union[str, Path, StringIO, BytesIO]): File path (str/Path) or in-memory stream (StringIO/BytesIO).
            sep (str): The delimiter character.
            header (bool): A boolean indicating if the data file has a header row.
            desired_cols (Optional[List[str]]): An optional list of column names to select.
            desired_dtypes (Optional[Dict[str, str]]): A dictionary mapping column names to desired data types.

        Returns:
            DataFrame[Any]: A Narwhals DataFrame containing the processed data.
        """
        if desired_dtypes is None:
            desired_dtypes = {}

        # Dispatch based on backend
        if self.backend == "polars":
            return self._read_with_polars(
                source, sep, header, desired_cols, desired_dtypes
            )
        return self._read_with_pandas(source, sep, header, desired_cols, desired_dtypes)

    def _read_with_pandas(
        self,
        source: Any,
        sep: str,
        header: bool,
        desired_cols: List[str] | None,
        desired_dtypes: Dict[str, str],
    ) -> DataFrame[Any]:
        """Internal method to read using Pandas."""

        # Helper for empty DF
        def _empty_nw_df(cols=None):
            cols = cols if cols is not None else []
            return nw.from_native(pd.DataFrame(columns=cols).astype(desired_dtypes))

        header_arg = 0 if header else None

        try:
            usecols = desired_cols if (header and desired_cols) else None

            # Basic read
            pd_df = pd.read_csv(
                source,
                sep=sep,
                header=header_arg,
                usecols=usecols,
                dtype=desired_dtypes
                if header
                else None,  # Dtypes apply easily if names match
            )

            if pd_df.empty:
                return _empty_nw_df(desired_cols)

        except pd.errors.EmptyDataError:
            return _empty_nw_df(desired_cols)
        except Exception as e:
            logger.negative(f"Error reading with Pandas: {e}")
            return _empty_nw_df(desired_cols)

        nw_df = nw.from_native(pd_df)

        # Post-processing for header=False or column filtering
        if not header and desired_cols:
            # Rename columns 0, 1, 2... to desired_cols
            current_cols = nw_df.columns
            num_cols = min(len(current_cols), len(desired_cols))
            rename_map = {current_cols[i]: desired_cols[i] for i in range(num_cols)}
            nw_df = nw_df.rename(rename_map)

        if desired_cols:
            # Filter columns (robust check)
            existing_cols = [c for c in desired_cols if c in nw_df.columns]
            if not existing_cols:
                return _empty_nw_df()

            # Select only desired columns after renaming
            nw_df = nw_df.select(existing_cols)

        return nw_df

    def _read_with_polars(
        self,
        source: Any,
        sep: str,
        header: bool,
        desired_cols: List[str] | None,
        desired_dtypes: Dict[str, str],
    ) -> DataFrame[Any]:
        """Internal method to read using Polars."""
        if isinstance(source, StringIO):
            source = BytesIO(source.getvalue().encode("utf-8"))

        try:
            columns_arg = desired_cols if (header and desired_cols) else None

            pl_df = pl.read_csv(
                source,
                separator=sep,
                has_header=header,
                columns=columns_arg,
                infer_schema_length=10000,
                truncate_ragged_lines=True,
            )

            if pl_df.height == 0:
                return nw.from_native(
                    pl.DataFrame(schema={c: pl.Utf8 for c in (desired_cols or [])})
                )

        except Exception as e:
            logger.negative(f"Error reading with Polars: {e}")
            return nw.from_native(pl.DataFrame())

        nw_df = nw.from_native(pl_df)

        # Post-processing
        if not header and desired_cols:
            current_cols = nw_df.columns
            num_cols = min(len(current_cols), len(desired_cols))
            rename_map = {current_cols[i]: desired_cols[i] for i in range(num_cols)}
            nw_df = nw_df.rename(rename_map)

        if desired_cols:
            existing_cols = [c for c in desired_cols if c in nw_df.columns]
            if not existing_cols:
                return nw.from_native(pl.DataFrame())
            nw_df = nw_df.select(existing_cols)

        return nw_df

    @abstractmethod
    def read_tabular(self, *args: Any, **kwargs: Any) -> DataFrame[Any]:
        """This method will read the tabular data from the source."""

    @abstractmethod
    def read_tabular_split(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[
        DataFrame[Any],
        Optional[List[Tuple[DataFrame[Any], DataFrame[Any]]] | DataFrame[Any]],
        DataFrame[Any],
    ]:
        """This method will read the tabular split data from the source."""

    def _process_tabular_split(
        self,
        base_location: str,
        column_names: Optional[List[str]] = None,
        dtypes: Optional[Dict[str, str]] = None,
        sep: str = "\t",
        ext: str = ".tsv",
        header: bool = True,
        is_remote: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[
        DataFrame[Any],
        Optional[List[Tuple[DataFrame[Any], DataFrame[Any]]] | DataFrame[Any]],
        DataFrame[Any],
    ]:
        """Reads split data (Train/Validation/Test)."""
        if dtypes is None:
            dtypes = {}

        path_joiner = posixpath.join if is_remote else lambda *a: str(Path(*a))

        path_main_train = path_joiner(base_location, f"train{ext}")
        path_main_val = path_joiner(base_location, f"validation{ext}")
        path_main_test = path_joiner(base_location, f"test{ext}")

        logger.msg(
            f"Starting reading split process from: {base_location} using {self.backend}"
        )

        train_data = self.read_tabular(
            path_main_train, column_names, dtypes, sep, header
        )
        test_data = self.read_tabular(path_main_test, column_names, dtypes, sep, header)

        # Narwhals check for empty
        if (
            train_data.select(nw.len()).item() == 0
            or test_data.select(nw.len()).item() == 0
        ):
            raise FileNotFoundError(
                f"Train/Test data not found or empty in '{base_location}'."
            )

        val_data = self.read_tabular(path_main_val, column_names, dtypes, sep, header)
        if val_data.select(nw.len()).item() > 0:
            return (train_data, val_data, test_data)

        # Iterate through fold subdirectories
        fold_data = []
        fold_number = 1
        while True:
            fold_path = path_joiner(base_location, str(fold_number))
            path_fold_train = path_joiner(fold_path, f"train{ext}")

            fold_train = self.read_tabular(
                path_fold_train, column_names, dtypes, sep, header
            )
            if fold_train.select(nw.len()).item() == 0:
                break

            path_fold_val = path_joiner(fold_path, f"validation{ext}")
            fold_val = self.read_tabular(
                path_fold_val, column_names, dtypes, sep, header
            )

            if fold_val.select(nw.len()).item() > 0:
                fold_data.append((fold_train, fold_val))
                fold_number += 1
            else:
                break

        logger.positive("Reading process completed successfully.")
        return (train_data, fold_data if fold_data else None, test_data)

    @abstractmethod
    def read_json(self, *args: Any, **kwargs: Any) -> DataFrame[Any]:
        """This method will read the json data from the source."""

    @abstractmethod
    def read_json_split(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[DataFrame[Any], DataFrame[Any] | None, DataFrame[Any] | None]:
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
        backend = config.general.backend

        # Create the appropriate Reader instance based on the reading method
        match reader_type:
            case ReadingMethods.LOCAL:
                from warprec.data.reader import LocalReader

                return LocalReader(backend=backend)
            case ReadingMethods.AZURE_BLOB:
                from warprec.data.reader import AzureBlobReader

                storage_account_name = config.general.azure.storage_account_name
                container_name = config.general.azure.container_name

                return AzureBlobReader(
                    storage_account_name=storage_account_name,
                    container_name=container_name,
                    backend=backend,
                )

        raise ValueError(f"Unknown reader type: {reader_type}")
