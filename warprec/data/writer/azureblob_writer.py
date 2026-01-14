# pylint: disable = R0801
import csv
import posixpath
from io import StringIO
from typing import Optional, Generator

from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobServiceClient

from warprec.data.writer import Writer
from warprec.data import Dataset
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.logger import logger


class AzureBlobWriter(Writer):
    """AzureBlobWriter saves the experiment results to Azure Blob Storage.

    Args:
        storage_account_name (str): The storage account name of the Azure Blob Storage.
        container_name (str): The name of the blob container where results will be stored.
        dataset_name (str): The name of the dataset.
        blob_experiment_container (str): The root container for experiments. Defaults to "experiments".
    """

    def __init__(
        self,
        storage_account_name: str,
        container_name: str,
        dataset_name: str,
        blob_experiment_container: str = "experiments",
    ):
        super().__init__()

        credential = DefaultAzureCredential()
        account_url = f"https://{storage_account_name}.blob.core.windows.net"
        self.blob_service_client = BlobServiceClient(
            account_url=account_url, credential=credential
        )
        self.container_name = container_name
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

        self.experiment_path = posixpath.join(blob_experiment_container, dataset_name)
        self.experiment_evaluation_path = posixpath.join(
            self.experiment_path, "evaluation"
        )
        self.experiment_recommendation_path = posixpath.join(
            self.experiment_path, "recs"
        )
        self.experiment_serialized_models_path = posixpath.join(
            self.experiment_path, "serialized"
        )
        self.experiment_params_path = posixpath.join(self.experiment_path, "params")
        self.experiment_split_path = posixpath.join(self.experiment_path, "split")

        self.setup_experiment()

    def _path_join(self, *args) -> str:
        """Joins path components for a blob URI."""
        return posixpath.join(*args)

    def setup_experiment(self):
        """Ensures the Azure Blob container exists."""
        logger.msg(
            f"Setting up experiment on Azure Blob container '{self.container_name}'."
        )
        try:
            self.container_client.create_container()
            logger.msg(f"Container '{self.container_name}' created.")
        except ResourceExistsError:
            logger.msg(f"Container '{self.container_name}' already exists.")
        logger.msg("Experiment setup on Azure completed successfully.")

    def write_recs(
        self,
        model: Recommender,
        dataset: Dataset,
        k: int,
        sep: str = "\t",
        ext: str = ".tsv",
        header: bool = True,
        user_label: str = "user_id",
        item_label: str = "item_id",
        rating_label: str = "rating",
    ) -> None:
        """Uploads recommendations to Azure Blob Storage in a streaming fashion."""
        path = self._path_join(
            self.experiment_recommendation_path,
            f"{model.name}_{self._timestamp}{ext}",
        )
        blob_client = self.container_client.get_blob_client(path)

        # Get the generator that yields batches of recommendation rows
        batch_generator = self._generate_recommendation_batches(model, dataset, k)

        def csv_batch_generator() -> Generator[bytes, None, None]:
            """A generator that yields CSV data in batches as UTF-8 encoded bytes."""
            string_io = StringIO()
            writer = csv.writer(string_io, delimiter=sep)

            if header:
                writer.writerow([user_label, item_label, rating_label])
                yield string_io.getvalue().encode("utf-8")
                string_io.seek(0)
                string_io.truncate(0)

            # Iterate over batches of rows and write them
            for batch in batch_generator:
                writer.writerows(batch)
                yield string_io.getvalue().encode("utf-8")

                # Reset the StringIO for the next batch
                string_io.seek(0)
                string_io.truncate(0)

            string_io.close()

        try:
            # upload_blob can take a generator for memory-efficient streaming
            blob_client.upload_blob(csv_batch_generator(), overwrite=True)
            logger.msg(f"Recommendations successfully written to blob: {path}")
        except AzureError as e:
            logger.negative(f"Error writing recommendations to blob {path}: {e}")

    def _write_text(self, path: str, content: str) -> None:
        """Uploads text content to a blob."""
        self.container_client.get_blob_client(path).upload_blob(
            content, overwrite=True, encoding="utf-8"
        )

    def _read_text(self, path: str) -> Optional[str]:
        """Downloads text content from a blob if it exists."""
        try:
            blob_client = self.container_client.get_blob_client(path)
            return blob_client.download_blob(encoding="utf-8").readall()
        except ResourceNotFoundError:
            return None

    def _write_bytes(self, path: str, content: bytes) -> None:
        """Uploads binary content to a blob."""
        self.container_client.get_blob_client(path).upload_blob(content, overwrite=True)
