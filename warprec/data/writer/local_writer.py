from os.path import join
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import json
import csv
from pandas import DataFrame
from torch import Tensor
from datetime import timedelta
from tqdm import tqdm

from warprec.data.writer.base_writer import Writer
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import (
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.utils.config import TrainConfiguration
from warprec.utils.enums import WritingMethods
from warprec.utils.config import (
    WriterConfig,
    ResultsWriting,
    SplitWriting,
    RecommendationWriting,
)
from warprec.utils.config.common import Labels
from warprec.utils.logger import logger


class LocalWriter(Writer):
    """LocalWriter is the class to be used when the results of
    the experiment want to be saved locally.

    Args:
        dataset_name (str): The name of the dataset.
        local_path (str): The path to the dataset.
        config (TrainConfiguration): The configuration of the experiment.
    """

    def __init__(
        self,
        dataset_name: str = None,
        local_path: str = None,
        config: TrainConfiguration = None,
    ):
        if config:
            self.config = config
            writer_params = config.writer
        else:
            # Setup experiment information from args
            writer_params = WriterConfig(
                dataset_name=dataset_name,
                writing_method=WritingMethods.LOCAL,
                local_experiment_path=local_path,
            )

        self._timestamp = datetime.now().strftime("%d.%m.%Y_%H:%M:%S")
        self.experiment_path = Path(
            join(
                writer_params.local_experiment_path,
                writer_params.dataset_name,
            )
        )
        self.experiment_evaluation_dir = Path(join(self.experiment_path, "evaluation"))
        self.experiment_recommendation_dir = Path(join(self.experiment_path, "recs"))
        self.experiment_serialized_models_dir = Path(
            join(self.experiment_path, "serialized")
        )
        self.experiment_params_dir = Path(join(self.experiment_path, "params"))
        self.experiment_split_dir = Path(join(self.experiment_path, "split"))

        # Setup the experimentation folder
        self.setup_experiment(config)

    def setup_experiment(self, config: TrainConfiguration = None):
        """This is the main function to be executed, it sets up all
        the important directory to then later save results."""
        logger.msg("Setting up experiment local folder.")

        # Check if directory exists and create the non existing one
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        self.experiment_evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_recommendation_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_serialized_models_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_params_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_split_dir.mkdir(parents=True, exist_ok=True)

        # Write locally the json file of the configuration
        if config:
            json_dump = config.model_dump_json(indent=2)
            json_path = Path(
                join(self.experiment_path, f"config_{self._timestamp}.json")
            )
            json_path.write_text(json_dump, encoding="utf-8")

        logger.msg("Experiment folder created successfully.")

    def write_results(
        self,
        result_data: Dict[int, Dict[str, float | Tensor]],
        model_name: str,
        sep: str = "\t",
        ext: str = ".tsv",
    ) -> None:
        """This function writes locally all the results of the experiment into a single
        "Overall_Results_{timestamp}.tsv" file, merging with existing data if present.

        Args:
            result_data (Dict[int, Dict[str, float | Tensor]]): The dictionary containing the results.
                Format: { "k": { "MetricName": value } }
                Example: {5: {"Precision": 0.1, "Recall": 0.2}}
            model_name (str): The name of the model which was evaluated.
            sep (str): The separator of the file.
            ext (str): The extension of the file.
        """
        if self.config:
            writing_params = self.config.writer.results
        else:
            writing_params = ResultsWriting(sep=sep, ext=ext)

        current_overall_results_path = Path(
            join(
                self.experiment_evaluation_dir,
                f"Overall_Results_{self._timestamp}{writing_params.ext}",
            )
        )

        # Load existing data if the file exists
        existing_df = pd.DataFrame()
        if current_overall_results_path.exists():
            try:
                existing_df = pd.read_csv(
                    current_overall_results_path, sep=writing_params.sep
                )
            except Exception as e:
                logger.attention(
                    f"Could not read existing overall results from {current_overall_results_path}: {e}. "
                    "A new file will be created or existing data will be overwritten."
                )
                existing_df = pd.DataFrame()  # Reset to empty if reading fails

        # Convert new results to a DataFrame
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

        # Save the combined DataFrame
        try:
            final_df.to_csv(
                current_overall_results_path, sep=writing_params.sep, index=False
            )
            logger.msg(
                f"Results successfully written in {current_overall_results_path}"
            )
        except Exception as e:
            logger.negative(
                f"Error writing results to {current_overall_results_path}: {e}"
            )

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
        """
        This method generates and writes recommendations to a file sequentially (batch by batch).

        Args:
            model (Recommender): The trained recommender model instance.
            dataset (Dataset): The dataset to use for retrieving data and mappings.
            k (int): The number of top-k recommendations to generate per user.
            sep (str): The separator for the output file.
            ext (str): The extension for the output file.
            header (bool): Whether to write the header in the file.
            user_label (str): The label for the user column.
            item_label (str): The label for the item column.
            rating_label (str): The label for the rating column.
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

        # Define the full output file path
        recommendation_file_path = join(
            self.experiment_recommendation_dir,
            f"{model.name}_{self._timestamp}{writing_params.ext}",
        )

        # Prepare data from the dataset
        train_sparse = dataset.train_set.get_sparse()
        umap_i, imap_i = dataset.get_inverse_mappings()
        num_users = train_sparse.shape[0]
        all_user_indices = torch.arange(num_users, device=model._device)
        batch_size = dataset._batch_size

        try:
            # Open the file in write mode ('w')
            with open(recommendation_file_path, "w", newline="", encoding="utf-8") as f:
                # Use the csv module for more robust writing
                writer = csv.writer(f, delimiter=writing_params.sep)

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

            logger.msg(
                f"Recommendations successfully written in {recommendation_file_path}"
            )

        except Exception as e:
            logger.negative(
                f"Error writing recommendations to {recommendation_file_path}: {e}"
            )

    def write_model(self, model: Recommender):
        """This method writes the model state into a local path.

        Args:
            model (Recommender): The model to write locally.
        """
        # experiment_path/serialized/model_name.pth
        _path = join(self.experiment_serialized_models_dir, model.name_param + ".pth")
        torch.save(model.state_dict(), _path)

    def write_params(self, params: dict) -> None:
        """This method writes the model parameters into a local path.

        Args:
            params (dict): The parameters of the model.

        NOTE: Params are expected to be a dictionary formatted as follows:
        {
            "model_name": {
                "param1": value1,
                "param2": value2,
                ...
            },
            ...
        }
        """
        # experiment_path/serialized/Overall_Params_{timestamp}.json
        _path = self.experiment_params_dir / f"Overall_Params_{self._timestamp}.json"

        existing_data = {}
        if _path.exists() and _path.stat().st_size > 0:
            try:
                with open(_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    logger.attention(
                        f"File {_path} does not contain a valid JSON object (dictionary). "
                        f"It will be overwritten with new and existing (if any) params."
                    )
                    # If it's not a dict, we can't update it.
                    # We'll effectively overwrite it, but first merge params into an empty dict
                    # to ensure params are definitely there.
                    existing_data = {}
            except json.JSONDecodeError:
                logger.attention(
                    f"Could not decode JSON from {_path}. The file will be overwritten "
                    f"with new and existing (if any) params."
                )
                existing_data = {}  # Reset on decode error
            except Exception as e:
                logger.negative(
                    f"Error reading {_path}: {e}. The file will be treated as empty/overwritten."
                )
                existing_data = {}

        # Update params with existing ones
        existing_data.update(params)

        try:
            with open(_path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=4)
            logger.msg(f"Parameters successfully written to {_path}")
        except Exception as e:
            logger.negative(f"Error writing parameters to {_path}: {e}")

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
        """This method writes the split into a local path.

        Args:
            main_dataset (Dataset): The main dataset split.
            val_dataset (Optional[Dataset]): The validation dataset split.
            fold_dataset (Optional[List[Dataset]]): The list of fold datasets.
            sep (str): The separator that will be used to write the results.
            ext (str): The extension that will be used to write the results.
            header (bool): Whether to write the header in the file.
            column_names (List[str] | None): Optional list of column names to use for the DataFrame.
                If None, the DataFrame's existing columns will be used.
        """

        # Helper function to write single dataset
        def write_dataset(dataset: Dataset, path: Path, eval_set: str):
            path_train = path.joinpath("train" + writing_params.ext)
            path_eval = path.joinpath(eval_set + writing_params.ext)

            df = dataset.train_set.get_df().copy()
            df = df[validated_column_names]
            df.to_csv(
                path_train,
                sep=writing_params.sep,
                header=writing_params.header,
                index=None,
            )

            df = dataset.eval_set.get_df().copy()
            df = df[validated_column_names]
            df.to_csv(
                path_eval,
                sep=writing_params.sep,
                header=writing_params.header,
                index=None,
            )

        if self.config:
            writing_params = self.config.writer.split
        else:
            if not column_names:
                column_names = main_dataset.train_set._inter_df.columns
            writing_params = SplitWriting(
                sep=sep, ext=ext, header=header, labels=Labels.from_list(column_names)
            )

        main_split_path = self.experiment_split_dir

        # Check the column to use
        infos = main_dataset.info()
        validated_column_names = [
            writing_params.labels.user_id_label,
            writing_params.labels.item_id_label,
        ]
        if infos["has_explicit_ratings"]:
            validated_column_names.append(writing_params.labels.rating_label)
        if infos["has_timestamp"]:
            validated_column_names.append(writing_params.labels.timestamp_label)

        write_dataset(main_dataset, main_split_path, "test")

        # If validation data is used, write it
        # in the main path
        if val_dataset is not None:
            write_dataset(val_dataset, main_split_path, "validation")

        # If fold data is used, we iterate over it and
        # write it locally
        if len(fold_dataset) > 0:
            for i, fold in enumerate(fold_dataset):
                fold_path = main_split_path.joinpath(str(i + 1))
                fold_path.mkdir(parents=True, exist_ok=True)
                write_dataset(fold, fold_path, "validation")

        logger.msg(f"Split data written to {main_split_path}")

    def write_time_report(self, time_report: List[Dict[str, Any]]):
        """This method writes the time report into a local path, with incremental updates.

        Args:
            time_report (List[Dict[str, Any]]): The time report to write.
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

        # experiment_path/evaluation/Time_Report_{timestamp}.{ext}
        time_report_path = Path(
            join(
                self.experiment_evaluation_dir,
                f"Time_Report_{self._timestamp}{writing_params.ext}",
            )
        )

        try:
            # Load existing data if the file exists
            existing_df = pd.DataFrame()
            if time_report_path.exists():
                try:
                    existing_df = pd.read_csv(time_report_path, sep=writing_params.sep)
                except Exception as e:
                    logger.attention(
                        f"Could not read existing time report from {time_report_path}: {e}. "
                        "A new file will be created or existing data will be overwritten."
                    )
                    existing_df = pd.DataFrame()

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

            report.to_csv(
                time_report_path,
                sep=writing_params.sep,
                index=False,
            )
            logger.msg(f"Time report written to {time_report_path}")
        except Exception as e:
            logger.negative(f"Error writing time report: {e}")

    def write_statistical_significance_test(
        self, test_results: DataFrame, test_name: str
    ):
        """This method writes the results of a statistical significance test into a local path.

        Args:
            test_results (DataFrame): The DataFrame containing the results of the statistical test.
            test_name (str): The name of the statistical test performed.
        """
        if self.config:
            writing_params = self.config.writer.results
        else:
            writing_params = ResultsWriting(sep="\t", ext=".tsv")

        # experiment_path/evaluation/{test_name}_{timestamp}.{ext}
        test_results_path = join(
            self.experiment_evaluation_dir,
            f"{test_name.capitalize()}_{self._timestamp}{writing_params.ext}",
        )
        try:
            test_results.to_csv(
                test_results_path,
                sep=writing_params.sep,
                index=False,
            )
            logger.msg(
                f"Statistical significance test results written to {test_results_path}"
            )
        except Exception as e:
            logger.negative(f"Error writing statistical significance test results: {e}")
