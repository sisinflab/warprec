from typing import Any, List, Tuple, Optional
from pathlib import Path

import pandas as pd
import joblib
from pandas import DataFrame

from warprec.data.reader.base_reader import Reader
from warprec.utils.config import (
    WarpRecConfiguration,
    ReaderConfig,
    Labels,
    CustomDtype,
    SplitReading,
    SideInformationReading,
    ClusteringInformationReading,
)
from warprec.utils.enums import RatingType, ReadingMethods
from warprec.utils.logger import logger


class LocalReader(Reader):
    """This class extends the AbstractReader, it handles
    all the data reading part from a local machine.

    Args:
        config (WarpRecConfiguration): Configuration file.
    """

    def __init__(self, config: WarpRecConfiguration = None) -> None:
        if config:
            self.config = config

    def _robust_read_csv(
        self,
        path: str | Path,
        sep: str,
        header: bool,
        desired_cols: List[str],
        desired_dtypes: dict,
    ) -> DataFrame:
        """Reads a CSV file robustly, handling missing columns and mismatched headers.

        Args:
            path (str | Path): The path to the local file.
            sep (str): The separator used in the file.
            header (bool): Whether the file has a header row.
            desired_cols (List[str]): The desired column names.
            desired_dtypes (dict): A dictionary mapping column names to their dtypes.

        Returns:
            DataFrame: The data read from the local source.

        Raises:
            FileNotFoundError: If the local path does not exists.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found at the specified path: {path}")

        cols_to_use = []
        dtype_to_use = {}

        if header:
            # Safe file reading
            try:
                file_cols = pd.read_csv(path, sep=sep, nrows=0).columns.tolist()
            except pd.errors.EmptyDataError:
                logger.attention(f"File {path} is empty. Returning an empty DataFrame.")
                return pd.DataFrame(columns=desired_cols).astype(desired_dtypes)

            # Filter out the correct columns that exist in the file
            cols_to_use = [col for col in desired_cols if col in file_cols]
            dtype_to_use = {
                col: desired_dtypes[col] for col in cols_to_use if col in desired_dtypes
            }

            if not cols_to_use:
                logger.attention(
                    "None of the desired columns were found in the file header. Returning an empty DataFrame."
                )
                return pd.DataFrame()

            # Read local file using correct information
            data = pd.read_csv(
                path,
                sep=sep,
                usecols=cols_to_use,
                dtype=dtype_to_use,
            )
        else:
            # Safe file reading
            try:
                data = pd.read_csv(path, sep=sep, header=None)
                if data.empty:
                    logger.attention(
                        f"File {path} is empty. Returning an empty DataFrame."
                    )
                    return pd.DataFrame(columns=desired_cols).astype(desired_dtypes)

            except pd.errors.EmptyDataError:
                logger.attention(f"File {path} is empty. Returning an empty DataFrame.")
                return pd.DataFrame(columns=desired_cols).astype(desired_dtypes)

            # Define the columns to use from local file
            num_cols_in_file = len(data.columns)
            num_cols_to_rename = min(num_cols_in_file, len(desired_cols))
            cols_to_use = desired_cols[:num_cols_to_rename]
            col_mapping = dict(zip(data.columns[:num_cols_to_rename], cols_to_use))

            # Rename and select the correct columns
            data.rename(columns=col_mapping, inplace=True)
            data = data[cols_to_use]

            # Cast the columns to the correct type
            dtype_to_use = {
                col: desired_dtypes[col] for col in cols_to_use if col in desired_dtypes
            }
            data = data.astype(dtype_to_use)

        return data

    def read(
        self,
        local_path: str = None,
        rating_type: RatingType = RatingType.IMPLICIT,
        sep: str = "\t",
        header: bool = True,
        column_names: List[str] | None = None,
        dtypes: List[str] | None = None,
        **kwargs: Any,
    ) -> DataFrame:
        """This method will read the data locally, using parameters from the config file.

        Args:
            local_path (str): The path to the local file.
            rating_type (RatingType): The rating type used in the dataset.
            sep (str): The separator used in the file.
            header (bool): Whether the file has a header row.
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
                header=header,
                rating_type=rating_type,
                labels=Labels.from_list(column_names),
                dtypes=CustomDtype(**dtypes_map),
            )

        logger.msg(
            f"Starting reading process from local source in: {read_config.local_path}"
        )

        data = self._robust_read_csv(
            path=read_config.local_path,
            sep=read_config.sep,
            header=read_config.header,
            desired_cols=read_config.column_names(),
            desired_dtypes=read_config.column_dtype(),
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
        header: bool = True,
        column_names: List[str] | None = None,
        dtypes: List[str] | None = None,
        rating_type: RatingType = RatingType.IMPLICIT,
        **kwargs: Any,
    ) -> Tuple[
        DataFrame, Optional[List[Tuple[DataFrame, DataFrame]] | DataFrame], DataFrame
    ]:
        """This method reads the split data from a local source.

        Args:
            split_dir (str): The path to the split directory.
            sep (str): The separator used in the file.
            ext (str): The extension of the split files.
            header (bool): Whether the split files have a header row.
            column_names (List[str] | None): The column names of the data.
            dtypes (List[str] | None): The data types of the columns.
            rating_type (RatingType): The type of rating to be used.
            **kwargs (Any): The keyword arguments.

        Returns:
            Tuple[DataFrame, Optional[List[Tuple[DataFrame, DataFrame]] | DataFrame], DataFrame]:
                - DataFrame: The main training set data.
                - List[Tuple[DataFrame, DataFrame]],: A list of tuples, where each tuple
                    contains the train and validation data for a single fold.
                - DataFrame: The main test set data.

        Raises:
            FileNotFoundError: If the main train split file was not found.
        """
        if self.config:
            read_config = self.config.reader
        else:
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
                split=SplitReading(
                    local_path=split_dir, ext=ext, sep=sep, header=header
                ),
            )

        path_split_dir = Path(read_config.split.local_path)

        # Define paths for the main split files
        path_main_train = path_split_dir.joinpath("train" + read_config.split.ext)
        path_main_val = path_split_dir.joinpath("validation" + read_config.split.ext)
        path_main_test = path_split_dir.joinpath("test" + read_config.split.ext)

        # Check for the existence of the main train file
        if not path_main_train.exists() or not path_main_test.exists():
            raise FileNotFoundError(
                "Split folder not correctly formatted. Train and "
                "Test data expected in the main folder."
            )

        logger.msg(f"Starting reading process from local source in: {path_split_dir}")

        # Read the main train and test data
        train_data = self._robust_read_csv(
            path=path_main_train,
            sep=read_config.side.sep,
            header=read_config.side.header,
            desired_cols=read_config.column_names(),
            desired_dtypes=read_config.column_dtype(),
        )
        test_data = self._robust_read_csv(
            path=path_main_test,
            sep=read_config.side.sep,
            header=read_config.side.header,
            desired_cols=read_config.column_names(),
            desired_dtypes=read_config.column_dtype(),
        )

        # Check if validation is in the main directory
        if path_main_val.exists():
            val_data = self._robust_read_csv(
                path=path_main_val,
                sep=read_config.side.sep,
                header=read_config.side.header,
                desired_cols=read_config.column_names(),
                desired_dtypes=read_config.column_dtype(),
            )
            return (train_data, val_data, test_data)

        # Iterate through subdirectories for folds
        fold_data = []
        fold_number = 1
        while True:
            fold_path = path_split_dir.joinpath(str(fold_number))

            # Check if the directory for the current fold number exists
            if fold_path.exists():
                # Paths for fold train and validation files
                path_fold_train = fold_path.joinpath("train" + read_config.split.ext)
                path_fold_val = fold_path.joinpath("validation" + read_config.split.ext)

                # Check if the required files exist inside the fold directory
                if path_fold_train.is_file() and path_fold_val.is_file():
                    fold_train = self._robust_read_csv(
                        path=path_fold_train,
                        sep=read_config.side.sep,
                        header=read_config.side.header,
                        desired_cols=read_config.column_names(),
                        desired_dtypes=read_config.column_dtype(),
                    )
                    fold_val = self._robust_read_csv(
                        path=path_fold_val,
                        sep=read_config.side.sep,
                        header=read_config.side.header,
                        desired_cols=read_config.column_names(),
                        desired_dtypes=read_config.column_dtype(),
                    )
                    fold_data.append((fold_train, fold_val))
                    fold_number += 1  # Move to the next fold number
                else:
                    break
            else:
                break

        logger.positive("Reading process completed successfully.")

        # Final check to conform to the return typing
        if len(fold_data) > 0:
            return (train_data, fold_data, test_data)
        return (train_data, None, test_data)

    def read_side_information(
        self,
        local_path: str = None,
        sep: str = "\t",
        header: bool = True,
    ) -> DataFrame:
        """This method will read the side information locally, using parameters from the config file.

        Args:
            local_path (str): The path to the local file.
            sep (str): The separator used in the file.
            header (bool): Whether the file has a header row.

        Returns:
            DataFrame: The data read from the local source.
        """
        if self.config:
            read_config = self.config.reader.side
        else:
            read_config = SideInformationReading(
                local_path=local_path, sep=sep, header=header
            )

        logger.msg(
            f"Reading side information from local source in: {read_config.local_path}"
        )
        if read_config.header:
            data = pd.read_csv(read_config.local_path, sep=read_config.sep)
        else:
            data = pd.read_csv(
                read_config.local_path,
                sep=read_config.sep,
                header=None,
            )
        logger.msg("Data loaded correctly from local source.")

        return data

    def read_cluster_information(
        self,
        user_local_path: str = None,
        item_local_path: str = None,
        user_sep: str = "\t",
        item_sep: str = "\t",
        user_header: bool = True,
        item_header: bool = True,
    ) -> Tuple[DataFrame, DataFrame]:
        """This method will read the cluster information locally, using parameters from the config file.

        Args:
            user_local_path (str): The path to the user cluster file.
            item_local_path (str): The path to the item cluster file.
            user_sep (str): The separator used in the user cluster file.
            item_sep (str): The separator used in the item cluster file.
            user_header (bool): Whether the user cluster file has a header row.
            item_header (bool): Whether the item cluster file has a header row.

        Returns:
            Tuple[DataFrame, DataFrame]: The user and item clusters data.
        """
        if self.config:
            read_config = self.config.reader.clustering
        else:
            read_config = ClusteringInformationReading(
                user_local_path=user_local_path,
                item_local_path=item_local_path,
                user_sep=user_sep,
                item_sep=item_sep,
                user_header=user_header,
                item_header=item_header,
            )

        # User clustering
        logger.msg(
            f"Reading user clustering information from local source in: {read_config.user_local_path}"
        )
        if read_config.user_header:
            user_data = pd.read_csv(
                read_config.user_local_path, sep=read_config.user_sep
            )
        else:
            user_data = pd.read_csv(
                read_config.user_local_path,
                sep=read_config.user_sep,
                header=None,
            )
        logger.msg("User data loaded correctly from local source.")

        # Item clustering
        logger.msg(
            f"Reading item clustering information from local source in: {read_config.item_local_path}"
        )
        if read_config.item_header:
            item_data = pd.read_csv(
                read_config.item_local_path, sep=read_config.item_sep
            )
        else:
            item_data = pd.read_csv(
                read_config.item_local_path,
                sep=read_config.item_sep,
                header=None,
            )
        logger.msg("Item data loaded correctly from local source.")

        return user_data, item_data
