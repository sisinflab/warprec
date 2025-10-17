from io import StringIO, BytesIO
from typing import List, Tuple, Optional

import pandas as pd
import joblib
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from pandas import DataFrame

from warprec.data.reader.base_reader import Reader
from warprec.utils.logger import logger


class AzureBlobReader(Reader):
    """This class extends Reader and handles data reading from an Azure Blob Storage container.

    It uses DefaultAzureCredential to authenticate, which relies on
    environment variables or other standard Azure identity sources.

    Args:
        storage_account_name (str): The name of the Azure Storage Account.
        container_name (str): The name of the container where data is stored.
    """

    def __init__(
        self,
        storage_account_name: str,
        container_name: str,
    ) -> None:
        # Retrieve Azure credentials from the environment
        credential = DefaultAzureCredential()

        # Create the BlobService client
        account_url = f"https://{storage_account_name}.blob.core.windows.net"
        self.blob_service_client = BlobServiceClient(
            account_url=account_url, credential=credential
        )

        # Retrieve the container client
        self.container_name = container_name
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )
        logger.msg(
            f"AzureBlobReader initialized for container '{self.container_name}'."
        )

    def _download_blob_content(
        self, blob_name: str, as_bytes: bool = False
    ) -> str | bytes | None:
        """Helper to download a blob's content.

        Args:
            blob_name (str): The full path/name of the blob within the container.
            as_bytes (bool): If True, returns content as raw bytes; otherwise, decodes to UTF-8 string.

        Returns:
            str | bytes | None: The blob content as a string or bytes,
                or None if the resource is not found.
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

    def read_tabular(
        self,
        blob_name: str,
        column_names: Optional[List[str]],
        dtypes: Optional[List[str]],
        sep: str = "\t",
        header: bool = True,
    ) -> DataFrame:
        """Reads tabular data from a blob by feeding it to the parent stream processor.

        Downloads the blob content as a string and uses the inherited `_process_csv_stream`
        for robust CSV parsing.

        Args:
            blob_name (str): The path/name of the blob containing the tabular data.
            column_names (Optional[List[str]]): A list of expected column names.
            dtypes (Optional[List[str]]): A list of data types corresponding to `column_names`.
            sep (str): The delimiter character used in the file. Defaults to tab `\t`.
            header (bool): A boolean indicating if the file has a header row. Defaults to `True`.

        Returns:
            DataFrame: A Pandas DataFrame containing the tabular data. Returns an empty DataFrame
                if the blob is not found.
        """
        content = self._download_blob_content(blob_name)
        if content is None:
            # Return an empty df that the split logic can check
            return pd.DataFrame()

        stream = StringIO(content)  # type: ignore[arg-type]

        # Create mapping only if values are not None
        dtypes_map = None
        if column_names and dtypes:
            dtypes_map = dict(zip(column_names, dtypes))

        return self._process_csv_stream(
            stream=stream,
            sep=sep,
            header=header,
            desired_cols=column_names,
            desired_dtypes=dtypes_map,
        )

    def read_tabular_split(  # type: ignore[override]
        self,
        blob_prefix: str,
        column_names: Optional[List[str]],
        dtypes: Optional[List[str]],
        sep: str = "\t",
        ext: str = ".tsv",
        header: bool = True,
    ) -> Tuple[
        DataFrame, Optional[List[Tuple[DataFrame, DataFrame]] | DataFrame], DataFrame
    ]:
        return super().read_tabular_split(
            base_location=blob_prefix,
            column_names=column_names,
            dtypes=dtypes,
            sep=sep,
            ext=ext,
            header=header,
            is_remote=True,  # Specify remote path handling
        )

    def read_json(self, *args, **kwargs):
        """This method will read the json data from the source."""
        raise NotImplementedError

    def read_json_split(self, *args, **kwargs):
        """This method will read the json split data from the source."""
        raise NotImplementedError

    def load_model_state(self, blob_name: str) -> dict:
        """This method will load a model state from a source.

        Downloads the blob content as bytes and uses `joblib.load` to deserialize the model state.

        Args:
            blob_name (str): The path/name of the blob containing the serialized model state (e.g., a `.joblib` file).

        Returns:
            dict: A dictionary representing the loaded model state (e.g., weights, hyperparameters).

        Raises:
            FileNotFoundError: If the model state blob is not found.
        """
        logger.msg(f"Loading model state from blob: {blob_name}")
        blob_content_bytes = self._download_blob_content(blob_name, as_bytes=True)

        if blob_content_bytes is None:
            raise FileNotFoundError(
                f"Model state blob '{blob_name}' not found in container '{self.container_name}'."
            )

        buffer = BytesIO(blob_content_bytes)  # type: ignore[arg-type]
        model_state = joblib.load(buffer)
        logger.msg("Model state loaded successfully.")
        return model_state
