import posixpath
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Dict, Union
from pathlib import Path
from io import StringIO, BytesIO

import numpy as np
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

    @abstractmethod
    def read_tabular(self, *args: Any, **kwargs: Any) -> DataFrame[Any]:
        """This method will read the tabular data from the source."""

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

        def _get_pandas_dtype(dtype_str: str) -> Any:
            mapping = {
                "int16": np.int16,
                "int32": np.int32,
                "int64": np.int64,
                "float32": np.float32,
                "float64": np.float64,
                "str": np.str_,
            }
            return mapping.get(dtype_str, "object")

        # Prepare dtype mapping
        pandas_dtypes = {}
        if desired_dtypes:
            for col_name, dtype_str in desired_dtypes.items():
                pandas_dtypes[col_name] = _get_pandas_dtype(dtype_str)

        try:
            # Case 1: The tabular file has a header row
            if header:
                # Peek at the header to validate columns
                # We use nrows=0 to just get the header
                if hasattr(source, "seek"):
                    source.seek(0)

                schema_df = pd.read_csv(source, sep=sep, header=0, nrows=0)
                file_cols = schema_df.columns.tolist()

                # Reset stream for the actual read
                if hasattr(source, "seek"):
                    source.seek(0)

                usecols = None
                if desired_cols:
                    # Intersect desired columns with those actually present
                    usecols = [col for col in desired_cols if col in file_cols]

                    if not usecols:
                        logger.attention(
                            "None of the desired columns were found in Pandas read. Returning empty."
                        )
                        # Return empty DF with correct schema
                        empty_df = pd.DataFrame(columns=desired_cols or [])
                        if pandas_dtypes:
                            # Apply dtypes to empty df to ensure schema correctness
                            valid_dtypes = {
                                k: v
                                for k, v in pandas_dtypes.items()
                                if k in empty_df.columns
                            }
                            empty_df = empty_df.astype(valid_dtypes)
                        return nw.from_native(empty_df)

                pd_df = pd.read_csv(
                    source,
                    sep=sep,
                    header=0,
                    usecols=usecols,
                    dtype=pandas_dtypes,
                )

            # Case 2: The tabular file does not have a header row
            else:
                # Read without header (columns will be 0, 1, 2...)
                pd_df = pd.read_csv(
                    source,
                    sep=sep,
                    header=None,
                )

                if pd_df.empty:
                    return nw.from_native(pd.DataFrame(columns=desired_cols or []))

                if desired_cols:
                    # Rename positional columns (0, 1...) to desired names
                    current_cols = pd_df.columns
                    num_cols = min(len(current_cols), len(desired_cols))
                    rename_map = {
                        current_cols[i]: desired_cols[i] for i in range(num_cols)
                    }

                    pd_df = pd_df.rename(columns=rename_map)

                    # Select only the renamed columns
                    pd_df = pd_df[list(rename_map.values())]

                    # Apply dtypes NOW, after renaming
                    if pandas_dtypes:
                        # Filter dtypes to only existing columns in the dataframe
                        valid_dtypes = {
                            k: v for k, v in pandas_dtypes.items() if k in pd_df.columns
                        }
                        try:
                            pd_df = pd_df.astype(valid_dtypes)
                        except Exception as e:
                            logger.negative(f"Error casting types in Pandas: {e}")

            if pd_df.empty:
                return nw.from_native(pd.DataFrame(columns=desired_cols or []))

        except pd.errors.EmptyDataError:
            return nw.from_native(pd.DataFrame(columns=desired_cols or []))
        except Exception as e:
            logger.negative(f"Error reading with Pandas: {e}")
            return nw.from_native(pd.DataFrame())

        nw_df = nw.from_native(pd_df)

        # Final check to ensure column order matches desired_cols if provided
        if desired_cols:
            existing_cols = [c for c in desired_cols if c in nw_df.columns]
            if existing_cols:
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

        def _get_polars_dtype(dtype_str: str) -> Any:
            mapping = {
                "int16": pl.Int16,
                "int32": pl.Int32,
                "int64": pl.Int64,
                "float32": pl.Float32,
                "float64": pl.Float64,
                "str": pl.String,
            }
            return mapping.get(dtype_str, pl.String)

        # Polars requires BytesIO for in-memory streams
        if isinstance(source, StringIO):
            source = BytesIO(source.getvalue().encode("utf-8"))

        # Parse the schema to override
        schema_overrides = {}
        if desired_dtypes:
            for col_name, dtype_str in desired_dtypes.items():
                schema_overrides[col_name] = _get_polars_dtype(dtype_str)

        try:
            # Case 1: The tabular file has a header row
            if header:
                # Peek at the header using n_rows=0
                schema_df = pl.read_csv(
                    source, separator=sep, has_header=True, n_rows=0
                )
                file_cols = schema_df.columns

                # Reset stream for the actual read
                if hasattr(source, "seek"):
                    source.seek(0)

                columns_arg = None
                if desired_cols:
                    # Intersect desired columns with those actually present
                    columns_arg = [col for col in desired_cols if col in file_cols]

                    if not columns_arg:
                        logger.attention(
                            "None of the desired columns were found. Returning empty."
                        )
                        return nw.from_native(
                            pl.DataFrame(schema={c: pl.Utf8 for c in desired_cols})
                        )

                pl_df = pl.read_csv(
                    source,
                    separator=sep,
                    has_header=True,
                    columns=columns_arg,
                    schema_overrides=schema_overrides,
                    infer_schema_length=10000,
                    truncate_ragged_lines=True,
                    rechunk=True,
                )

            # Case 2: The tabular file does not have a header row
            else:
                pl_df = pl.read_csv(
                    source,
                    separator=sep,
                    has_header=False,
                    schema_overrides=schema_overrides,
                    infer_schema_length=10000,
                    truncate_ragged_lines=True,
                    rechunk=True,
                )

                if pl_df.height == 0:
                    return nw.from_native(
                        pl.DataFrame(schema={c: pl.Utf8 for c in (desired_cols or [])})
                    )

                if desired_cols:
                    # Rename positional columns (column_1, column_2...)
                    current_cols = pl_df.columns
                    num_cols = min(len(current_cols), len(desired_cols))
                    rename_map = {
                        current_cols[i]: desired_cols[i] for i in range(num_cols)
                    }

                    pl_df = pl_df.rename(rename_map)
                    # Select only the renamed columns
                    pl_df = pl_df.select(list(rename_map.values()))

            if pl_df.height == 0:
                return nw.from_native(
                    pl.DataFrame(schema={c: pl.Utf8 for c in (desired_cols or [])})
                )

        except Exception as e:
            logger.negative(f"Error reading with Polars: {e}")
            return nw.from_native(pl.DataFrame())

        nw_df = nw.from_native(pl_df)

        # Final check to ensure column order matches desired_cols if provided
        if desired_cols:
            existing_cols = [c for c in desired_cols if c in nw_df.columns]
            if existing_cols:
                nw_df = nw_df.select(existing_cols)

        return nw_df

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
    def read_parquet(self, *args: Any, **kwargs: Any) -> DataFrame[Any]:
        """This method will read the parquet data from the source."""

    def _process_parquet_data(
        self,
        source: Union[str, Path, BytesIO],
        desired_cols: Optional[List[str]] = None,
    ) -> DataFrame[Any]:
        """Internal method to process parquet data based on the selected backend.

        Args:
            source (Union[str, Path, BytesIO]): File path or stream.
            desired_cols (Optional[List[str]]): List of columns to read.

        Returns:
            DataFrame[Any]: A Narwhals DataFrame.
        """
        try:
            if self.backend == "polars":
                # Polars read_parquet
                pl_df = pl.read_parquet(source, columns=desired_cols)
                return nw.from_native(pl_df)
            else:
                # Pandas read_parquet
                pd_df = pd.read_parquet(source, columns=desired_cols)
                return nw.from_native(pd_df)
        except Exception as e:
            logger.negative(f"Error reading Parquet with {self.backend}: {e}")
            return nw.from_native(pd.DataFrame())

    @abstractmethod
    def read_parquet_split(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[
        DataFrame[Any],
        Optional[List[Tuple[DataFrame[Any], DataFrame[Any]]] | DataFrame[Any]],
        DataFrame[Any],
    ]:
        """This method will read the parquet split data from the source."""

    def _process_parquet_split(
        self,
        base_location: str,
        column_names: Optional[List[str]] = None,
        ext: str = ".parquet",
        is_remote: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[
        DataFrame[Any],
        Optional[List[Tuple[DataFrame[Any], DataFrame[Any]]] | DataFrame[Any]],
        DataFrame[Any],
    ]:
        """Reads split data (Train/Validation/Test) from Parquet files."""

        path_joiner = posixpath.join if is_remote else lambda *a: str(Path(*a))

        path_main_train = path_joiner(base_location, f"train{ext}")
        path_main_val = path_joiner(base_location, f"validation{ext}")
        path_main_test = path_joiner(base_location, f"test{ext}")

        logger.msg(
            f"Starting reading parquet split process from: {base_location} using {self.backend}"
        )

        train_data = self.read_parquet(path_main_train, column_names)
        test_data = self.read_parquet(path_main_test, column_names)

        if (
            train_data.select(nw.len()).item() == 0
            or test_data.select(nw.len()).item() == 0
        ):
            raise FileNotFoundError(
                f"Train/Test parquet data not found or empty in '{base_location}'."
            )

        val_data = self.read_parquet(path_main_val, column_names)

        if val_data.select(nw.len()).item() > 0:
            return (train_data, val_data, test_data)

        # Iterate over the folds
        fold_data = []
        fold_number = 1
        while True:
            fold_path = path_joiner(base_location, str(fold_number))
            path_fold_train = path_joiner(fold_path, f"train{ext}")

            fold_train = self.read_parquet(path_fold_train, column_names)

            if fold_train.select(nw.len()).item() == 0:
                break

            path_fold_val = path_joiner(fold_path, f"validation{ext}")
            fold_val = self.read_parquet(path_fold_val, column_names)

            if fold_val.select(nw.len()).item() > 0:
                fold_data.append((fold_train, fold_val))
                fold_number += 1
            else:
                break

        logger.positive("Parquet reading process completed successfully.")
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
