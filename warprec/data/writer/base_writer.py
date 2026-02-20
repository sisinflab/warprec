import json
from typing import Dict, Optional, List, Any, Generator
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from io import StringIO, BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from narwhals.dataframe import DataFrame
from torch import Tensor
from tqdm import tqdm

from warprec.data import Dataset
from warprec.recommenders.base_recommender import (
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.utils.config import TrainConfiguration, EvalConfiguration
from warprec.utils.enums import WritingMethods
from warprec.utils.logger import logger


class Writer(ABC):
    """Writer is the base definition of a writer that centralizes processing logic
    and delegates I/O operations to subclasses.
    """

    experiment_path: str | Path
    experiment_evaluation_path: str | Path
    experiment_recommendation_path: str | Path
    experiment_serialized_models_path: str | Path
    experiment_params_path: str | Path
    experiment_split_path: str | Path

    def __init__(self):
        self._timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

    @abstractmethod
    def setup_experiment(self):
        """Sets up the storage environment (e.g., creates folders or containers)."""

    @abstractmethod
    def _write_text(self, path: str, content: str):
        """Writes a text string to a specific destination."""

    @abstractmethod
    def _read_text(self, path: str) -> Optional[str]:
        """Reads a text string from a destination. Returns None if it doesn't exist."""

    @abstractmethod
    def _write_bytes(self, path: str, content: bytes):
        """Writes binary data to a specific destination."""

    @abstractmethod
    def _path_join(self, *args: Any) -> str:
        """Joins path components using the correct separator for the platform."""

    @abstractmethod
    def write_recs(self, *args: Any, **kwargs: Any):
        """
        Consumes the recommendation generator and writes the data to the destination
        in a memory-efficient, streaming fashion.
        """

    def _df_to_csv_string(
        self, df: DataFrame[Any], sep: str = "\t", header: bool = True
    ) -> str:
        """Helper method to convert a DataFrame to a CSV string
        using Narwhals for backend detection.

        Args:
            df (DataFrame[Any]): The DataFrame to write.
            sep (str): The separator to use.
            header (bool): Wether or not to save with header.

        Returns:
            str: The converted DataFrame to string.

        Raises:
            ValueError: If the DataFrame is in a format not supported.
        """
        native_df = df.to_native()

        if isinstance(native_df, pl.DataFrame):
            return native_df.write_csv(None, separator=sep, include_header=header)

        if isinstance(native_df, pd.DataFrame):
            return native_df.to_csv(sep=sep, header=header, index=False)

        raise ValueError(
            "The DataFrame is in a format not compatible with the writer. "
            f"DataFrame type: {type(native_df)}"
        )

    def _df_to_parquet_bytes(self, df: DataFrame[Any]) -> bytes:
        """Helper method to convert a DataFrame to Parquet bytes.

        Args:
            df (DataFrame[Any]): The DataFrame to write.

        Returns:
            bytes: The DataFrame converted to parquet bytes.

        Raises:
            ValueError: If the DataFrame type is not supported.
        """
        native_df = df.to_native()
        buffer = BytesIO()

        if isinstance(native_df, pl.DataFrame):
            native_df.write_parquet(buffer)
        elif isinstance(native_df, pd.DataFrame):
            native_df.to_parquet(buffer, index=False)
        else:
            raise ValueError(
                "The DataFrame is in a format not compatible with the writer. "
                f"DataFrame type: {type(native_df)}"
            )

        return buffer.getvalue()

    def _generate_recommendation_batches(
        self, model: Recommender, dataset: Dataset, k: int
    ) -> Generator[list[tuple], None, None]:
        """A generator that yields batches of recommendation rows.
        Each batch corresponds to the recommendations for a batch of users.

        Args:
            model (Recommender): The trained model from which to produce recommendations.
            dataset (Dataset): The dataset used to train the model.
            k (int): The number of recommendations to produce for each user.

        Yields:
            list[tuple]: A list of (user_label, item_label, score) tuples.
        """
        train_sparse = dataset.train_set.get_sparse()
        umap_i, imap_i = dataset.get_inverse_mappings()
        num_users = train_sparse.shape[0]
        all_user_indices = torch.arange(num_users, device=model.device)
        batch_size = dataset.batch_size

        batch_iterator = range(0, num_users, batch_size)
        for i in tqdm(batch_iterator, desc="Generating recommendation batches"):
            user_indices = all_user_indices[i : i + batch_size]
            train_batch = train_sparse[user_indices.tolist(), :]

            user_seq, seq_len = (
                dataset.train_session.get_user_history_sequences(
                    user_indices.tolist(), model.max_seq_len
                )
                if isinstance(model, SequentialRecommenderUtils)
                else (None, None)
            )

            with torch.inference_mode():
                predictions = model.predict(
                    user_indices=user_indices,
                    user_seq=user_seq,
                    seq_len=seq_len,
                )
                predictions[train_batch.nonzero()] = -torch.inf
                top_k_scores, top_k_items = torch.topk(predictions, k, dim=1)

            batch_users = user_indices.unsqueeze(1).expand(-1, k).flatten()
            user_labels = [umap_i[idx.item()] for idx in batch_users]
            item_labels = [imap_i[idx.item()] for idx in top_k_items.flatten()]
            scores = top_k_scores.flatten().tolist()

            # Yield the entire list of rows for the current batch
            yield list(zip(user_labels, item_labels, scores))

    def write_results(
        self,
        result_data: Dict[int, Dict[str, float | Tensor]],
        model_name: str,
        sep: str = "\t",
        ext: str = ".tsv",
    ) -> None:
        """Process and write results of the experiment.

        Args:
            result_data (Dict[int, Dict[str, float | Tensor]]): The results
                of the experiment.
            model_name (str): The model used to produce the results.
            sep (str): The separator of the file to produce.
            ext (str): The extension of the file to produce.
        """
        # This implementation remains the same as before
        path = self._path_join(
            self.experiment_evaluation_path,
            f"Overall_Results_{self._timestamp}{ext}",
        )
        existing_content = self._read_text(path)
        existing_df = pd.DataFrame()
        if existing_content:
            try:
                existing_df = pd.read_csv(StringIO(existing_content), sep=sep)
            except (pd.errors.ParserError, ValueError, pd.errors.EmptyDataError) as e:
                logger.attention(
                    f"Could not parse existing results from {path}: {e}. Overwriting."
                )
        new_result_list = []
        for k, metrics in result_data.items():
            row = {"Model": model_name, "Top@k": k}
            for metric_name, metric_result in metrics.items():
                value = (
                    metric_result.nanmean().item()
                    if isinstance(metric_result, Tensor)
                    else metric_result
                )
                row[metric_name] = value
            new_result_list.append(row)
        new_df = pd.DataFrame(new_result_list)
        merge_keys = ["Model", "Top@k"]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        metric_columns = sorted(combined_df.columns.difference(merge_keys))
        final_column_order = merge_keys + metric_columns
        final_df = (
            combined_df.drop_duplicates(subset=merge_keys, keep="last")
            .reindex(columns=final_column_order)
            .sort_values(by=merge_keys)
            .reset_index(drop=True)
        )
        try:
            output_csv = final_df.to_csv(sep=sep, index=False)
            self._write_text(path, output_csv)
            logger.msg(f"Results successfully written to {path}")
        except (pd.errors.ParserError, ValueError, pd.errors.EmptyDataError) as e:
            logger.negative(f"Error writing results to {path}: {e}")

    def write_results_per_user(
        self,
        result_data: Dict[int, Dict[str, float | Tensor]],
        model_name: str,
        user_mapping: Dict[int, Any],
        sep: str = "\t",
        ext: str = ".tsv",
        user_label: str = "user_id",
    ):
        """Processes and writes per-user evaluation metrics.

        This method handles the data transformation logic and delegates the actual
        I/O to the `_write_text` method, which must be implemented by a subclass.

        Args:
            result_data (Dict[int, Dict[str, float | Tensor]]): Dictionary containing per-user
                result data.
            model_name (str): The name of the model used.
            user_mapping (Dict[int, Any]): The dictionary that maps the idx -> ID.
            sep (str): The separator of the output file.
            ext (str): The extension of the output file.
            user_label (str): The label to use for the user ID.
        """
        for k, metrics in result_data.items():
            path = self._path_join(
                self.experiment_evaluation_path,
                f"{model_name}_k_{k}_per_user_{self._timestamp}{ext}",
            )

            try:
                # Filter out possible non Tensor metrics
                metric_tensors = {
                    name: tensor
                    for name, tensor in metrics.items()
                    if isinstance(tensor, Tensor)
                }

                if not metric_tensors:
                    logger.attention(
                        f"No per-user tensor metrics found for model {model_name} at k={k}. Skipping."
                    )
                    continue

                # Determine number of users by tensor length
                num_users = len(next(iter(metric_tensors.values())))

                # Pre-compute the user column
                user_ids = [user_mapping[i] for i in range(num_users)]

                # Prepare data
                data_for_df: Dict[str, Any] = {user_label: user_ids}
                for name, tensor in metric_tensors.items():
                    data_for_df[name] = tensor.cpu().numpy()

                # Create result DataFrame
                df = pd.DataFrame(data_for_df)
                output_csv = df.to_csv(sep=sep, index=False)

                # Finalize the writing
                self._write_text(path, output_csv)

                logger.msg(
                    f"Per-user results for {model_name} (k={k}) written to {path}"
                )

            except (pd.errors.ParserError, ValueError, pd.errors.EmptyDataError) as e:
                logger.negative(
                    f"Error writing per-user results for {model_name} (k={k}) to {path}: {e}"
                )

    def write_model(self, model: Recommender):
        """Saves the model's state dictionary.

        Args:
            model (Recommender): The model to write.
        """
        path = self._path_join(
            self.experiment_serialized_models_path, model.name_param + ".pth"
        )
        try:
            buffer = BytesIO()
            torch.save(model.get_state(), buffer)
            buffer.seek(0)
            self._write_bytes(path, buffer.read())
            logger.msg(f"Model state successfully written to {path}")
        except (pd.errors.ParserError, ValueError, pd.errors.EmptyDataError) as e:
            logger.negative(f"Error writing model to {path}: {e}")

    def write_params(self, params: dict) -> None:
        """Writes model parameters, handling merging with existing data."""
        path = self._path_join(
            self.experiment_params_path, f"Overall_Params_{self._timestamp}.json"
        )

        existing_data = {}
        existing_content = self._read_text(path)
        if existing_content:
            try:
                existing_data = json.loads(existing_content)
                if not isinstance(existing_data, dict):
                    existing_data = {}
            except json.JSONDecodeError:
                logger.attention(
                    f"Could not decode JSON from {path}. It will be overwritten."
                )
                existing_data = {}

        existing_data.update(params)

        try:
            output_json = json.dumps(existing_data, indent=4)
            self._write_text(path, output_json)
            logger.msg(f"Parameters successfully written to {path}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.negative(f"Error writing parameters to {path}: {e}")

    def write_tabular_split(
        self,
        main_dataset: Dataset,
        val_dataset: Optional[Dataset],
        fold_dataset: Optional[List[Dataset]],
        sep: str = "\t",
        ext: str = ".tsv",
        header: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        """Writes the dataset splits in tabular format (CSV/TSV)."""

        def _write_dataset(dataset: Dataset, path_prefix: str, eval_set_name: str):
            try:
                # Write train set
                train_path = self._path_join(path_prefix, f"train{ext}")
                train_csv = self._df_to_csv_string(
                    dataset.train_set.get_df(), sep=sep, header=header
                )
                self._write_text(train_path, train_csv)

                # Write eval set
                eval_path = self._path_join(path_prefix, f"{eval_set_name}{ext}")
                eval_csv = self._df_to_csv_string(
                    dataset.eval_set.get_df(), sep=sep, header=header
                )
                self._write_text(eval_path, eval_csv)
            except Exception as e:
                logger.negative(f"Failed to write tabular split to {path_prefix}: {e}")

        main_split_path = str(self.experiment_split_path)

        # Main Split
        _write_dataset(main_dataset, main_split_path, "test")

        # Validation Split (if exists)
        if val_dataset:
            _write_dataset(val_dataset, main_split_path, "validation")

        # Folds (if exist)
        if fold_dataset:
            for i, fold in enumerate(fold_dataset):
                fold_path = self._path_join(main_split_path, str(i + 1))
                _write_dataset(fold, fold_path, "validation")

        logger.msg(f"Tabular split data written to {main_split_path}")

    def write_parquet_split(
        self,
        main_dataset: Dataset,
        val_dataset: Optional[Dataset],
        fold_dataset: Optional[List[Dataset]],
        ext: str = ".parquet",
        *args: Any,
        **kwargs: Any,
    ):
        """Writes the dataset splits in Parquet format."""

        def _write_dataset(dataset: Dataset, path_prefix: str, eval_set_name: str):
            try:
                # Write train set
                train_path = self._path_join(path_prefix, f"train{ext}")
                train_bytes = self._df_to_parquet_bytes(dataset.train_set.get_df())
                self._write_bytes(train_path, train_bytes)

                # Write eval set
                eval_path = self._path_join(path_prefix, f"{eval_set_name}{ext}")
                eval_bytes = self._df_to_parquet_bytes(dataset.eval_set.get_df())
                self._write_bytes(eval_path, eval_bytes)
            except Exception as e:
                logger.negative(f"Failed to write parquet split to {path_prefix}: {e}")

        main_split_path = str(self.experiment_split_path)

        # Main Split
        _write_dataset(main_dataset, main_split_path, "test")

        # Validation Split (if exists)
        if val_dataset:
            _write_dataset(val_dataset, main_split_path, "validation")

        # Folds (if exist)
        if fold_dataset:
            for i, fold in enumerate(fold_dataset):
                fold_path = self._path_join(main_split_path, str(i + 1))
                _write_dataset(fold, fold_path, "validation")

        logger.msg(f"Parquet split data written to {main_split_path}")

    def write_time_report(
        self, time_report: List[Dict[str, Any]], sep: str = "\t", ext: str = ".tsv"
    ):
        """Writes the time report, handling merging with existing data."""
        path = self._path_join(
            self.experiment_evaluation_path,
            f"Time_Report_{self._timestamp}{ext}",
        )

        existing_content = self._read_text(path)
        existing_df = pd.DataFrame()
        if existing_content:
            try:
                existing_df = pd.read_csv(StringIO(existing_content), sep=sep)
            except (pd.errors.ParserError, ValueError, pd.errors.EmptyDataError) as e:
                logger.attention(
                    f"Could not parse existing time report from {path}: {e}. Overwriting."
                )

        new_df = pd.DataFrame(time_report)
        new_df["Inference Time (ms)"] = (new_df.pop("Inference Time") * 1000).round(6)
        for col in new_df.select_dtypes(include=["float"]).columns:
            if "Usage (MB)" in col:
                new_df[col] = new_df[col].round(6)
            else:
                new_df[col] = new_df[col].apply(
                    lambda s: str(timedelta(seconds=s)) if pd.notna(s) else np.nan
                )

        merge_keys = ["Model Name"]
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        report = combined_df.drop_duplicates(subset=merge_keys, keep="last")

        first_columns = [
            "Model Name",
            "Trainable Params (Best Model)",
            "Total Params (Best Model)",
        ]
        other_cols = [col for col in report.columns if col not in first_columns]
        report = (
            report[first_columns + other_cols]
            .sort_values(by=merge_keys)
            .reset_index(drop=True)
        )

        try:
            output_csv = report.to_csv(sep=sep, index=False)
            self._write_text(path, output_csv)
            logger.msg(f"Time report written to {path}")
        except (pd.errors.ParserError, ValueError, pd.errors.EmptyDataError) as e:
            logger.negative(f"Error writing time report to {path}: {e}")

    def write_statistical_significance_test(
        self,
        test_results: DataFrame,
        test_name: str,
        sep: str = "\t",
        ext: str = ".tsv",
    ) -> None:
        """Writes the results of a statistical significance test."""
        path = self._path_join(
            self.experiment_evaluation_path,
            f"{test_name.capitalize()}_{self._timestamp}{ext}",
        )
        try:
            output_csv = self._df_to_csv_string(test_results, sep=sep)
            self._write_text(path, output_csv)
            logger.msg(f"Statistical significance test results written to {path}")
        except (pd.errors.ParserError, ValueError, pd.errors.EmptyDataError) as e:
            logger.negative(f"Error writing statistical test results to {path}: {e}")


class WriterFactory:  # pylint: disable=C0415, R0903
    """Factory class for creating Writer instances based on configuration.

    Attributes:
        config (TrainConfiguration | EvalConfiguration): The configuration of the experiment.
    """

    config: TrainConfiguration | EvalConfiguration = None

    @classmethod
    def get_writer(cls, config: TrainConfiguration | EvalConfiguration) -> Writer:
        """Factory method to get the appropriate Writer instance based on the configuration.

        Args:
            config (TrainConfiguration | EvalConfiguration): The configuration of the experiment.

        Returns:
            Writer: An instance of a class that extends the Writer abstract class.

        Raises:
            ValueError: If the writing method specified in the configuration is unknown.
        """
        writer_type = config.writer.writing_method

        # Create the appropriate Writer instance based on the writing method
        match writer_type:
            case WritingMethods.LOCAL:
                from warprec.data.writer import LocalWriter

                dataset_name = config.writer.dataset_name
                local_path = config.writer.local_experiment_path

                return LocalWriter(
                    dataset_name=dataset_name,
                    local_path=local_path,
                )
            case WritingMethods.AZURE_BLOB:
                from warprec.data.writer import AzureBlobWriter

                storage_account_name = config.general.azure.storage_account_name
                container_name = config.general.azure.container_name
                dataset_name = config.writer.dataset_name
                blob_experiment_container = (
                    config.writer.azure_blob_experiment_container
                )

                return AzureBlobWriter(
                    storage_account_name=storage_account_name,
                    container_name=container_name,
                    dataset_name=dataset_name,
                    blob_experiment_container=blob_experiment_container,
                )

        raise ValueError(f"Unknown writer type: {writer_type}")
