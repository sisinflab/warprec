import csv
import json
import posixpath
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from azure.storage.blob import BlobServiceClient
from pandas import DataFrame
from torch import Tensor
from tqdm import tqdm

from warprec.data.writer import Writer
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import (
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.utils.enums import WritingMethods
from warprec.utils.config import (
    WriterConfig,
    ResultsWriting,
    SplitWriting,
    RecommendationWriting,
)
from warprec.utils.config.common import Labels
from warprec.utils.config import TrainConfiguration
from warprec.utils.logger import logger


class AzureBlobWriter(Writer):
    """AzureBlobWriter is the class to be used when the results of
    the experiment want to be saved on Azure Blob Storage.

    Args:
        connection_string (str): The connection string for the Azure Storage Account.
        container_name (str): The name of the blob container where results will be stored.
        dataset_name (str): The name of the dataset.
        blob_experiment_container (str): The Azure Blob container name. Defaults to "experiments".
        config (TrainConfiguration): The configuration of the experiment.
    """

    def __init__(
        self,
        connection_string: str,
        container_name: str,
        dataset_name: str = None,
        blob_experiment_container: str = "experiments",
        config: TrainConfiguration = None,
    ):
        # TODO: Write correct config
        if config:
            self.config = config
            writer_params = config.writer
        else:
            # Setup experiment information from args
            writer_params = WriterConfig(
                dataset_name=dataset_name,
                writing_method=WritingMethods.AZURE_BLOB,
                azure_blob_experiment_container=blob_experiment_container,
            )

        # Init Azure Blob client
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self.container_name = container_name
        self.container_client = self.blob_service_client.get_container_client(
            container_name
        )

        self._timestamp = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
        self.experiment_path = posixpath.join(
            writer_params.azure_blob_experiment_container, writer_params.dataset_name
        )
        self.experiment_evaluation_dir = posixpath.join(
            self.experiment_path, "evaluation"
        )
        self.experiment_recommendation_dir = posixpath.join(
            self.experiment_path, "recs"
        )
        self.experiment_serialized_models_dir = posixpath.join(
            self.experiment_path, "serialized"
        )
        self.experiment_params_dir = posixpath.join(self.experiment_path, "params")
        self.experiment_split_dir = posixpath.join(self.experiment_path, "split")

        # Setup the experimentation container
        self.setup_experiment(config)

    def setup_experiment(self, config: TrainConfiguration = None):
        """This function sets up the experiment, ensuring the container exists
        and writing the configuration file.
        """
        logger.msg(
            f"Setting up experiment on Azure Blob container '{self.container_name}'."
        )

        # Creating the container if it does not exist
        try:
            self.container_client.create_container()
            logger.msg(f"Container '{self.container_name}' created.")
        except ResourceExistsError:  # In this case, the container already exists
            logger.msg(f"Container '{self.container_name}' already exists.")

        # Upload config file inside the container
        if config:
            json_dump = config.model_dump_json(indent=2)
            blob_name = posixpath.join(
                self.experiment_path, f"config_{self._timestamp}.json"
            )
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(json_dump, overwrite=True, encoding="utf-8")

        logger.msg("Experiment setup on Azure completed successfully.")

    def _download_blob_to_dataframe(self, blob_name: str, sep: str) -> DataFrame:
        """Helper function to download a blob and read it into a Pandas DataFrame.

        Args:
            blob_name (str): The name of the blob to download.
            sep (str): The separator used in the CSV file.

        Returns:
            DataFrame: The DataFrame containing the blob data, or an empty DataFrame if
                          the blob does not exist or cannot be read.
        """
        try:
            # Download the blob content and read into a DataFrame
            blob_client = self.container_client.get_blob_client(blob_name)
            downloader = blob_client.download_blob(encoding="utf-8")
            blob_content = downloader.readall()
            return pd.read_csv(StringIO(blob_content), sep=sep)
        except ResourceNotFoundError:
            # No file found, return empty DataFrame
            return DataFrame()
        except Exception as e:
            logger.attention(
                f"Could not read existing blob {blob_name}: {e}. Treating as empty."
            )
            return DataFrame()

    def write_results(
        self,
        result_data: Dict[int, Dict[str, float | Tensor]],
        model_name: str,
        sep: str = "\t",
        ext: str = ".tsv",
    ) -> None:
        """This function writes experiment results to a single blob,
        merging with existing data if present.

        Args:
            result_data (Dict[int, Dict[str, float | Tensor]]): The dictionary containing the results.
                Format: { "k": { "MetricName": value } }
                Example: {5: {"Precision": 0.1, "Recall": 0.2}}
            model_name (str): The name of the model.
            sep (str): The separator to use in the output file.
            ext (str): The file extension to use for the output file.
        """
        if self.config:
            writing_params = self.config.writer.results
        else:
            writing_params = ResultsWriting(sep=sep, ext=ext)

        blob_name = posixpath.join(
            self.experiment_evaluation_dir,
            f"Overall_Results_{self._timestamp}{writing_params.ext}",
        )

        # Download existing results if the blob exists
        existing_df = self._download_blob_to_dataframe(blob_name, writing_params.sep)

        # Format new results into a DataFrame
        new_result_list = []
        for k_value, metrics in result_data.items():
            row = {"Model": model_name, "Top@k": k_value}
            for metric_name, metric_result in metrics.items():
                value = (
                    metric_result.mean().item()
                    if isinstance(metric_result, Tensor)
                    else metric_result
                )
                row.update({metric_name: value})
                new_result_list.append(row)

        new_df = pd.DataFrame(new_result_list)

        # Predefined columns used also as merge keys
        merge_keys = ["Model", "Top@k"]

        # If existing_df is empty, concat will just return new_df
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Drop duplicates based on merge keys, keeping the last occurrence
        final_df = combined_df.drop_duplicates(subset=merge_keys, keep="last")

        # Ensure the correct ordering of columns
        metric_columns = sorted(final_df.columns.difference(merge_keys))
        final_column_order = merge_keys + metric_columns

        # Reindex to ensure the final DataFrame has the correct column order
        # and sort by merge keys
        final_df = (
            combined_df.drop_duplicates(subset=merge_keys, keep="last")
            .reindex(columns=final_column_order)
            .sort_values(by=merge_keys)
            .reset_index(drop=True)
        )

        # Upload the merged results back to the blob
        try:
            output = final_df.to_csv(sep=writing_params.sep, index=False)
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(output, overwrite=True, encoding="utf-8")
            logger.msg(f"Results successfully written to blob: {blob_name}")
        except Exception as e:
            logger.negative(f"Error writing results to blob {blob_name}: {e}")

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
        """Generates and writes recommendations to a blob by processing in batches
        and uploading the final result.

        Args:
            model (Recommender): The trained recommender model.
            dataset (Dataset): The dataset to generate recommendations for.
            k (int): The number of top recommendations to generate per user.
            sep (str): The separator to use in the output file.
            ext (str): The file extension to use for the output file.
            header (bool): Whether to include a header row in the output file.
            user_label (str): The label to use for the user ID column.
            item_label (str): The label to use for the item ID column.
            rating_label (str): The label to use for the rating column.
        """
        if self.config:
            writing_params = self.config.writer.recommendation
        else:
            writing_params = RecommendationWriting(
                sep=sep,
                ext=ext,
                header=header,
                user_label=user_label,
                item_label=item_label,
                rating_label=rating_label,
            )

        blob_name = posixpath.join(
            self.experiment_recommendation_dir,
            f"{model.name}_{self._timestamp}{writing_params.ext}",
        )

        train_sparse = dataset.train_set.get_sparse()
        umap_i, imap_i = dataset.get_inverse_mappings()
        num_users = train_sparse.shape[0]
        all_user_indices = torch.arange(num_users, device=model._device)
        batch_size = dataset._batch_size

        try:
            # We use a buffer to accumulate CSV data before uploading
            string_io = StringIO()
            writer = csv.writer(string_io, delimiter=writing_params.sep)

            # Write the header once at the beginning, if requested
            if writing_params.header:
                writer.writerow(
                    [
                        writing_params.user_label,
                        writing_params.item_label,
                        writing_params.rating_label,
                    ]
                )

            # Main loop for batch processing with tqdm
            batch_iterator = range(0, num_users, batch_size)
            for i in tqdm(batch_iterator, desc="Processing recommendation batches"):
                user_indices = all_user_indices[i : i + batch_size]
                train_batch = train_sparse[user_indices.tolist(), :]
                user_seq, seq_len = None, None
                if isinstance(model, SequentialRecommenderUtils):
                    user_seq, seq_len = (
                        dataset.train_session.get_user_history_sequences(
                            user_indices.tolist(),
                            model.max_seq_len,
                        )
                    )

                # Generate predictions for the current batch
                predictions = model.predict_full(
                    user_indices=user_indices,
                    user_seq=user_seq,
                    seq_len=seq_len,
                    train_batch=train_batch,
                    train_sparse=train_sparse,
                )

                predictions[train_batch.nonzero()] = -torch.inf
                top_k_scores, top_k_items = torch.topk(predictions, k, dim=1)

                # Prepare the data for writing
                batch_users = user_indices.unsqueeze(1).expand(-1, k).flatten()
                top_k_items = top_k_items.flatten()
                top_k_scores = top_k_scores.flatten()

                # Map indices back to original labels
                user_labels = [umap_i[idx.item()] for idx in batch_users]
                item_labels = [imap_i[idx.item()] for idx in top_k_items]
                scores = top_k_scores.tolist()

                # Combine data into rows and write them to the file
                rows_to_write = zip(user_labels, item_labels, scores)
                writer.writerows(rows_to_write)

            # Upload the accumulated data to the blob
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                string_io.getvalue(), overwrite=True, encoding="utf-8"
            )
            logger.msg(f"Recommendations successfully written to blob: {blob_name}")

        except Exception as e:
            logger.negative(f"Error writing recommendations to blob {blob_name}: {e}")
        finally:
            string_io.close()

    def write_model(self, model: Recommender):
        """Writes the model state to a blob.

        Args:
            model (Recommender): The model to be saved.
        """
        blob_name = posixpath.join(
            self.experiment_serialized_models_dir, f"{model.name_param}.pth"
        )

        try:
            # Save the model state to a BytesIO buffer
            buffer = BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)  # Rewind the buffer to the beginning

            # Upload the buffer content to the blob
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(buffer.read(), overwrite=True)
            logger.msg(f"Model state successfully written to blob: {blob_name}")
        except Exception as e:
            logger.negative(f"Error writing model to blob {blob_name}: {e}")

    def write_params(self, params: dict) -> None:
        """Writes model parameters to a JSON blob, merging with existing data.

        Args:
            params (dict): The parameters to be saved.
        """
        blob_name = posixpath.join(
            self.experiment_params_dir, f"Overall_Params_{self._timestamp}.json"
        )
        blob_client = self.container_client.get_blob_client(blob_name)

        existing_data = {}
        try:
            if blob_client.exists():
                downloader = blob_client.download_blob(encoding="utf-8")
                existing_data = json.loads(downloader.readall())
                if not isinstance(existing_data, dict):
                    logger.attention(
                        f"Blob {blob_name} does not contain a valid JSON object. It will be overwritten."
                    )
                    existing_data = {}
        except json.JSONDecodeError:
            logger.attention(
                f"Could not decode JSON from {blob_name}. It will be overwritten."
            )
            existing_data = {}
        except Exception as e:
            logger.negative(
                f"Error reading blob {blob_name}: {e}. It will be treated as empty."
            )
            existing_data = {}

        existing_data.update(params)

        try:
            # Upload the merged parameters back to the blob
            output = json.dumps(existing_data, indent=4)
            blob_client.upload_blob(output, overwrite=True, encoding="utf-8")
            logger.msg(f"Parameters successfully written to blob: {blob_name}")
        except Exception as e:
            logger.negative(f"Error writing parameters to blob {blob_name}: {e}")

    def write_split(
        self,
        main_dataset: Dataset,
        val_dataset: Optional[Dataset],
        fold_dataset: Optional[List[Dataset]],
        sep: str = "\t",
        ext: str = ".tsv",
        header: bool = True,
        column_names: List[str] | None = None,
    ) -> None:
        """Writes the dataset splits into blobs.

        Args:
            main_dataset (Dataset): The main dataset containing train and test splits.
            val_dataset (Optional[Dataset]): The validation dataset, if any.
            fold_dataset (Optional[List[Dataset]]): List of fold datasets for cross-validation, if any.
            sep (str): Separator for the output files.
            ext (str): File extension for the output files.
            header (bool): Whether to include a header row in the output files.
            column_names (List[str] | None): List of column names to use. If None, defaults will be used.
        """

        def write_dataset_to_blob(dataset: Dataset, path_prefix: str, eval_set: str):
            """Helper function to write a single dataset split to Azure."""
            try:
                # Upload the training set
                train_blob_name = posixpath.join(
                    path_prefix, f"train{writing_params.ext}"
                )
                df_train = dataset.train_set.get_df().copy()[validated_column_names]
                output_train = df_train.to_csv(
                    sep=writing_params.sep, header=writing_params.header, index=None
                )
                self.container_client.get_blob_client(train_blob_name).upload_blob(
                    output_train, overwrite=True, encoding="utf-8"
                )

                # Upload the evaluation set (test or validation)
                eval_blob_name = posixpath.join(
                    path_prefix, f"{eval_set}{writing_params.ext}"
                )
                df_eval = dataset.eval_set.get_df().copy()[validated_column_names]
                output_eval = df_eval.to_csv(
                    sep=writing_params.sep, header=writing_params.header, index=None
                )
                self.container_client.get_blob_client(eval_blob_name).upload_blob(
                    output_eval, overwrite=True, encoding="utf-8"
                )
            except Exception as e:
                logger.negative(f"Failed to write dataset split to {path_prefix}: {e}")

        if self.config:
            writing_params = self.config.writer.split
        else:
            if not column_names:
                column_names = main_dataset.train_set._inter_df.columns
            writing_params = SplitWriting(
                sep=sep, ext=ext, header=header, labels=Labels.from_list(column_names)
            )

        infos = main_dataset.info()
        validated_column_names = [
            writing_params.labels.user_id_label,
            writing_params.labels.item_id_label,
        ]
        if infos["has_explicit_ratings"]:
            validated_column_names.append(writing_params.labels.rating_label)
        if infos["has_timestamp"]:
            validated_column_names.append(writing_params.labels.timestamp_label)

        # Write the dataset to blobs
        write_dataset_to_blob(main_dataset, self.experiment_split_dir, "test")
        if val_dataset is not None:
            write_dataset_to_blob(val_dataset, self.experiment_split_dir, "validation")
        if len(fold_dataset) > 0:
            for i, fold in enumerate(fold_dataset):
                fold_prefix = posixpath.join(self.experiment_split_dir, str(i + 1))
                write_dataset_to_blob(fold, fold_prefix, "validation")

        logger.msg(
            f"Split data written to Azure Blob prefix: {self.experiment_split_dir}"
        )

    def write_time_report(self, time_report: List[Dict[str, Any]]):
        """Writes the time report to a blob, with incremental updates.

        Args:
            time_report (List[Dict[str, Any]]): List of time report entries.
        """

        def format_secs(secs):
            try:
                return str(timedelta(seconds=secs))
            except Exception:
                return np.nan

        if self.config:
            writing_params = self.config.writer.results
        else:
            writing_params = ResultsWriting(sep="\t", ext=".tsv")

        blob_name = posixpath.join(
            self.experiment_evaluation_dir,
            f"Time_Report_{self._timestamp}{writing_params.ext}",
        )

        # Download existing time report if the blob exists
        existing_df = self._download_blob_to_dataframe(blob_name, writing_params.sep)

        # Convert new results to a DataFrame
        new_df = pd.DataFrame(time_report)

        # Inference time conversion for cleaner output
        new_df["Inference Time"] = (new_df["Inference Time"] * 1000).round(6)
        new_df = new_df.rename(columns={"Inference Time": "Inference Time (ms)"})

        # Rounding memory usage values
        new_df["RAM Mean Usage (MB)"] = new_df["RAM Mean Usage (MB)"].round(6)
        new_df["RAM STD Usage (MB)"] = new_df["RAM STD Usage (MB)"].round(6)
        new_df["RAM Max Usage (MB)"] = new_df["RAM Max Usage (MB)"].round(6)
        new_df["RAM Min Usage (MB)"] = new_df["RAM Min Usage (MB)"].round(6)
        new_df["VRAM Mean Usage (MB)"] = new_df["VRAM Mean Usage (MB)"].round(6)
        new_df["VRAM STD Usage (MB)"] = new_df["VRAM STD Usage (MB)"].round(6)
        new_df["VRAM Max Usage (MB)"] = new_df["VRAM Max Usage (MB)"].round(6)
        new_df["VRAM Min Usage (MB)"] = new_df["VRAM Min Usage (MB)"].round(6)

        # Now, proceed with formatting and reordering as in your original method
        float_columns = new_df.select_dtypes(include=["float32", "float64"]).columns
        columns_to_exclude = [
            "RAM Mean Usage (MB)",
            "RAM STD Usage (MB)",
            "RAM Max Usage (MB)",
            "RAM Min Usage (MB)",
            "VRAM Mean Usage (MB)",
            "VRAM STD Usage (MB)",
            "VRAM Max Usage (MB)",
            "VRAM Min Usage (MB)",
            "Inference Time (ms)",
        ]
        columns_to_format = [
            col for col in float_columns if col not in columns_to_exclude
        ]

        new_df = new_df.copy()
        for col in columns_to_format:
            new_df[col] = new_df[col].apply(format_secs)

        # Merge the new data with the existing data
        # 'Model Name' is the key to identify unique reports for a model
        merge_keys = ["Model Name"]

        # Concat the two dataframes and drop duplicates based on merge keys, keeping the last (newest) data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        report = combined_df.drop_duplicates(subset=merge_keys, keep="last")

        # Reordering columns
        first_columns = [
            "Model Name",
            "Trainable Params (Best Model)",
            "Total Params (Best Model)",
        ]
        other_cols = [col for col in report.columns if col not in first_columns]
        report = report[first_columns + other_cols]

        # Sort the final dataframe by the Model Name for consistency
        report = report.sort_values(by=merge_keys).reset_index(drop=True)

        try:
            # Upload the merged time report back to the blob
            output = report.to_csv(sep=writing_params.sep, index=False)
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(output, overwrite=True, encoding="utf-8")
            logger.msg(f"Time report written to blob: {blob_name}")
        except Exception as e:
            logger.negative(f"Error writing time report to blob {blob_name}: {e}")

    def write_statistical_significance_test(
        self, test_results: DataFrame, test_name: str
    ):
        """Writes the results of a statistical significance test to a blob.

        Args:
            test_results (DataFrame): The results of the statistical test.
            test_name (str): The name of the statistical test.
        """
        if self.config:
            writing_params = self.config.writer.results
        else:
            writing_params = ResultsWriting(sep="\t", ext=".tsv")

        blob_name = posixpath.join(
            self.experiment_evaluation_dir,
            f"{test_name.capitalize()}_{self._timestamp}{writing_params.ext}",
        )
        try:
            # Upload the statistical test results to the blob
            output = test_results.to_csv(sep=writing_params.sep, index=False)
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.upload_blob(output, overwrite=True, encoding="utf-8")
            logger.msg(f"Statistical test results written to blob: {blob_name}")
        except Exception as e:
            logger.negative(
                f"Error writing statistical test results to blob {blob_name}: {e}"
            )
