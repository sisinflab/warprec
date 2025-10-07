import posixpath
from io import StringIO, BytesIO
from typing import Any, List, Tuple, Optional

import pandas as pd
import joblib
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from pandas import DataFrame

from warprec.utils.config import (
    WarpRecConfiguration,
    ReaderConfig,
    Labels,
    CustomDtype,
    SplitReading,
    SideInformationReading,
    ClusteringInformationReading,
    AzureConfig,
)
from warprec.utils.enums import RatingType, ReadingMethods
from warprec.utils.logger import logger
from warprec.data.reader import Reader


class AzureBlobReader(Reader):
    """This class extends the Reader ABC, it handles all the data
    reading part from an Azure Blob Storage container.

    Args:
        storage_account_name (str): The storage account name of the Azure Blob Storage.
        container_name (str): The name of the blob container where data is stored.
        config (WarpRecConfiguration): Configuration file.
    """

    def __init__(
        self,
        storage_account_name: str,
        container_name: str,
        config: WarpRecConfiguration = None,
    ) -> None:
        if config:
            self.config = config
            azure_config = config.azure
        else:
            azure_config = AzureConfig(
                storage_account_name=storage_account_name,
                container_name=container_name,
            )

        # Retrieve Azure credentials from environment
        credential = DefaultAzureCredential()

        # Create the BlobService client
        account_url = (
            f"https://{azure_config.storage_account_name}.blob.core.windows.net"
        )
        self.blob_service_client = BlobServiceClient(
            account_url=account_url, credential=credential
        )

        # Retrieve the container client
        self.container_name = azure_config.container_name
        self.container_client = self.blob_service_client.get_container_client(
            azure_config.container_name
        )

        logger.msg(
            f"AzureBlobReader initialized for container '{self.container_name}'."
        )

    def _download_blob_content(
        self, blob_name: str, as_bytes: bool = False
    ) -> str | bytes | None:
        """Helper to download a blob's content.

        Args:
            blob_name (str): The name of the blob to download.
            as_bytes (bool): Whether to return the content as bytes. Defaults to False.

        Returns:
            str | bytes | None: The content as a string or bytes, or None if not found.
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            downloader = blob_client.download_blob()
            if as_bytes:
                return downloader.readall()
            return downloader.readall().decode("utf-8")
        except ResourceNotFoundError:
            logger.attention(
                f"Blob '{blob_name}' not found in container '{self.container_name}'."
            )
            return None

    def read(
        self,
        blob_name: str = None,
        rating_type: RatingType = RatingType.IMPLICIT,
        sep: str = "\t",
        header: bool = True,
        column_names: Optional[List[str]] = None,
        dtypes: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> DataFrame:
        """This method will read the data from a blob, using parameters from the config file.

        Args:
            blob_name (str): The name of the blob to read.
            rating_type (RatingType): The type of rating to consider.
            sep (str): The separator used in the blob file.
            header (bool): Whether the blob file has a header row.
            column_names (Optional[List[str]]): List of column names to use.
            dtypes (Optional[List[str]]): List of data types corresponding to column names.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            DataFrame: The loaded data as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the specified blob is not found.
            ValueError: If the blob content is not a valid string.
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
                loading_strategy="dataset",
                data_type="transaction",
                reading_method=ReadingMethods.AZURE_BLOB,
                azure_blob_name=blob_name,
                sep=sep,
                header=header,
                rating_type=rating_type,
                labels=Labels.from_list(column_names),
                dtypes=CustomDtype(**dtypes_map),
            )

        blob_to_read = blob_name or read_config.azure_blob_name
        logger.msg(f"Starting reading process from blob: {blob_to_read}")

        blob_content = self._download_blob_content(blob_to_read)
        if blob_content is None:
            raise FileNotFoundError(
                f"Blob '{blob_to_read}' not found in container '{self.container_name}'."
            )

        if not isinstance(blob_content, str):
            raise ValueError(
                f"Blob content for '{blob_to_read}' is not a valid string."
            )

        buffer = StringIO(blob_content)

        if read_config.header:
            data = pd.read_csv(
                buffer,
                sep=read_config.sep,
                usecols=read_config.column_names(),
                dtype=read_config.column_dtype(),
            )
        else:
            data = pd.read_csv(buffer, sep=read_config.sep, header=None)
            data.columns = read_config.column_names()

        logger.msg(f"Data loaded correctly from blob: {blob_to_read}.")
        return data

    def load_model_state(self, blob_name: str = None, **kwargs: Any) -> dict:
        """This method will load a model state from a given blob.

        Args:
            blob_name (str): The name of the blob containing the model state.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            dict: The loaded model state.

        Raises:
            ValueError: If no blob name is provided.
            FileNotFoundError: If the specified blob is not found.
        """
        if not blob_name:
            raise ValueError("Blob name for the model state must be provided.")

        logger.msg(f"Loading model state from blob: {blob_name}")

        blob_content_bytes = self._download_blob_content(blob_name, as_bytes=True)

        if blob_content_bytes is None:
            raise FileNotFoundError(
                f"Model state blob '{blob_name}' not found in container '{self.container_name}'."
            )

        if not isinstance(blob_content_bytes, bytes):
            raise ValueError(f"Blob content for '{blob_name}' is not valid bytes.")

        buffer = BytesIO(blob_content_bytes)
        model_state = joblib.load(buffer)
        logger.msg("Model state loaded successfully.")
        return model_state

    def read_transaction_split(
        self,
        blob_prefix: str = None,
        sep: str = "\t",
        ext: str = ".tsv",
        header: bool = True,
        column_names: Optional[List[str]] = None,
        dtypes: Optional[List[str]] = None,
        rating_type: RatingType = RatingType.IMPLICIT,
        **kwargs: Any,
    ) -> Tuple[
        DataFrame, Optional[List[Tuple[DataFrame, DataFrame]] | DataFrame], DataFrame
    ]:
        """This method reads split data from a specified prefix in Azure Blob Storage.

        Args:
            blob_prefix (str): The prefix in the blob storage where split files are located.
            sep (str): The separator used in the files.
            ext (str): The file extension of the split files.
            header (bool): Whether the files have a header row.
            column_names (Optional[List[str]]): List of column names to use.
            dtypes (Optional[List[str]]): List of data types corresponding to column names.
            rating_type (RatingType): The type of rating to consider.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[DataFrame, Optional[List[Tuple[DataFrame, DataFrame]] | DataFrame], DataFrame]: The
                training data, optional validation data (either as a list of folds or a single DataFrame),
                and test data.

        Raises:
            FileNotFoundError: If the expected train and test files are not found.
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
                reading_method=ReadingMethods.AZURE_BLOB,
                rating_type=rating_type,
                labels=Labels.from_list(column_names),
                dtypes=CustomDtype(**dtypes_map),
                split=SplitReading(
                    azure_blob_prefix=blob_prefix, ext=ext, sep=sep, header=header
                ),
            )

        def read_blob_to_df(blob_name: str) -> DataFrame | None:
            content = self._download_blob_content(blob_name)
            if content is None:
                return None  # Blob not found
            if not isinstance(content, str):
                return None  # Should not happen, fallback to None
            buffer = StringIO(content)
            if read_config.split.header:
                return pd.read_csv(
                    buffer,
                    sep=read_config.split.sep,
                    usecols=read_config.column_names(),
                    dtype=read_config.column_dtype(),
                )
            else:
                df = pd.read_csv(buffer, sep=read_config.split.sep, header=None)
                df.columns = read_config.column_names()
                return df.astype(read_config.column_dtype())

        prefix = blob_prefix or read_config.split.azure_blob_prefix

        blob_main_train = posixpath.join(prefix, f"train{read_config.split.ext}")
        blob_main_val = posixpath.join(prefix, f"validation{read_config.split.ext}")
        blob_main_test = posixpath.join(prefix, f"test{read_config.split.ext}")

        logger.msg(f"Starting reading split from Azure Blob prefix: {prefix}")

        train_data = read_blob_to_df(blob_main_train)
        test_data = read_blob_to_df(blob_main_test)

        if train_data is None or test_data is None:
            raise FileNotFoundError(
                f"Train and Test data expected in blob prefix '{prefix}'."
            )

        val_data = read_blob_to_df(blob_main_val)
        if val_data is not None:
            return (train_data, val_data, test_data)

        fold_data = []
        fold_number = 1
        while True:
            fold_prefix = posixpath.join(prefix, str(fold_number))
            blob_fold_train = posixpath.join(
                fold_prefix, f"train{read_config.split.ext}"
            )
            blob_fold_val = posixpath.join(
                fold_prefix, f"validation{read_config.split.ext}"
            )

            fold_train_df = read_blob_to_df(blob_fold_train)
            fold_val_df = read_blob_to_df(blob_fold_val)

            if fold_train_df is not None and fold_val_df is not None:
                fold_data.append((fold_train_df, fold_val_df))
                fold_number += 1
            else:
                break

        logger.positive("Reading process completed successfully.")

        return (train_data, fold_data if fold_data else None, test_data)

    def read_side_information(
        self,
        blob_name: str = None,
        sep: str = "\t",
        header: bool = True,
    ) -> DataFrame:
        """This method will read the side information from a blob.

        Args:
            blob_name (str): The name of the blob to read.
            sep (str): The separator used in the blob file.
            header (bool): Whether the blob file has a header row.

        Returns:
            DataFrame: The loaded side information as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the side information blob is not found.
            ValueError: If the blob content is not a valid string.
        """
        if self.config:
            read_config = self.config.reader.side
        else:
            read_config = SideInformationReading(
                azure_blob_name=blob_name, sep=sep, header=header
            )

        blob_to_read = blob_name or read_config.azure_blob_name
        logger.msg(f"Reading side information from blob: {blob_to_read}")

        content = self._download_blob_content(blob_to_read)
        if content is None:
            raise FileNotFoundError(
                f"Side information blob '{blob_to_read}' not found."
            )

        if not isinstance(content, str):
            raise ValueError(
                f"Blob content for '{blob_to_read}' is not a valid string."
            )

        buffer = StringIO(content)
        if read_config.header:
            data = pd.read_csv(buffer, sep=read_config.sep)
        else:
            data = pd.read_csv(buffer, sep=read_config.sep, header=None)

        logger.msg("Side information loaded correctly.")
        return data

    def read_cluster_information(
        self,
        user_blob_name: str = None,
        item_blob_name: str = None,
        user_sep: str = "\t",
        item_sep: str = "\t",
        user_header: bool = True,
        item_header: bool = True,
    ) -> Tuple[DataFrame, DataFrame]:
        """This method will read the cluster information from blobs.

        Args:
            user_blob_name (str): The name of the blob containing user clustering information.
            item_blob_name (str): The name of the blob containing item clustering information.
            user_sep (str): The separator used in the user clustering blob file.
            item_sep (str): The separator used in the item clustering blob file.
            user_header (bool): Whether the user clustering blob file has a header row.
            item_header (bool): Whether the item clustering blob file has a header row.

        Returns:
            Tuple[DataFrame, DataFrame]: The user and item clustering data as pandas DataFrames
        """
        if self.config:
            read_config = self.config.reader.clustering
        else:
            read_config = ClusteringInformationReading(
                user_azure_blob_name=user_blob_name,
                item_azure_blob_name=item_blob_name,
                user_sep=user_sep,
                item_sep=item_sep,
                user_header=user_header,
                item_header=item_header,
            )

        # Helper for reading a single cluster blob
        def read_cluster_blob(blob_name: str, sep: str, header: bool) -> DataFrame:
            logger.msg(f"Reading cluster information from blob: {blob_name}")
            content = self._download_blob_content(blob_name)
            if content is None:
                raise FileNotFoundError(f"Cluster blob '{blob_name}' not found.")
            if not isinstance(content, str):
                raise ValueError(
                    f"Blob content for '{blob_name}' is not a valid string."
                )

            buffer = StringIO(content)
            if header:
                return pd.read_csv(buffer, sep=sep)
            else:
                return pd.read_csv(buffer, sep=sep, header=None)

        # Read user clustering
        user_blob = user_blob_name or read_config.user_azure_blob_name
        user_data = read_cluster_blob(
            user_blob, read_config.user_sep, read_config.user_header
        )
        logger.msg("User cluster data loaded correctly.")

        # Read item clustering
        item_blob = item_blob_name or read_config.item_azure_blob_name
        item_data = read_cluster_blob(
            item_blob, read_config.item_sep, read_config.item_header
        )
        logger.msg("Item cluster data loaded correctly.")

        return user_data, item_data
