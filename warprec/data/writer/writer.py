from os.path import join
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import torch
import json
from pandas import DataFrame
from torch import Tensor
from datetime import timedelta

from warprec.utils.config import TrainConfiguration
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.enums import WritingMethods
from warprec.utils.config import (
    WriterConfig,
    ResultsWriting,
    SplitWriting,
    RecommendationWriting,
)
from warprec.utils.config.common import Labels
from warprec.utils.logger import logger


class Writer(ABC):
    """Writer is the abstract definition of a writer,
    during an experiment multiple writers can be defined.

    Attributes:
        config (TrainConfiguration): The configuration of the experiment.

    TODO: Use Factory Pattern for different writer.
    """

    config: TrainConfiguration = None

    @abstractmethod
    def setup_experiment(self):
        """This is the main function to be executed, it sets up all
        the important directory to then later save results.
        """

    @abstractmethod
    def write_results(
        self,
        result_dict: Dict[int, Dict[str, float | Tensor]],
        model_name: str,
    ):
        """This function writes all the results of the experiment."""

    @abstractmethod
    def write_recs(self, recs: DataFrame, model_name: str):
        """This method writes recommendations in the destination."""

    @abstractmethod
    def write_model(self, model: Recommender):
        """This method writes the model state into a destination."""

    @abstractmethod
    def write_split(self, main_dataset: Dataset, fold_dataset: Optional[List[Dataset]]):
        """This method writes the split of the dataset into a destination."""


class LocalWriter(Writer):
    """LocalWriter is the class to be used when the results of
    the experiment want to be saved locally.

    Args:
        dataset_name (str): The name of the dataset.
        local_path (str): The path to the dataset.
        config (TrainConfiguration): The configuration of the experiment.
        setup (bool): Flag value for the setup of the experiment.

    TODO: Using context manager
    """

    def __init__(
        self,
        dataset_name: str = None,
        local_path: str = None,
        config: TrainConfiguration = None,
        setup: bool = True,
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
                setup_experiment=setup,
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

        if writer_params.setup_experiment:
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
        recs: DataFrame,
        model_name: str,
        sep: str = "\t",
        ext: str = ".tsv",
        header: bool = False,
        user_label: str = "user_id",
        item_label: str = "item_id",
        rating_label: str = "rating",
    ) -> None:
        """This method writes recommendations in the local path.

        Args:
            recs (DataFrame): The recommendations in DataFrame format.
            model_name (str): The name of the model which produced the recommendations.
            sep (str): The separator of the file.
            ext (str): The extension of the file.
            header (bool): Whether to write the header in the file.
            user_label (str): The label of the user data.
            item_label (str): The label of the item data.
            rating_label (str): The label of the rating data.
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

        # experiment_path/recs/model_name.{custom_extension}
        recommendation_folder_path = join(
            self.experiment_recommendation_dir,
            f"{model_name}_{self._timestamp}{writing_params.ext}",
        )

        try:
            # Save in path
            recs.to_csv(
                recommendation_folder_path,
                sep=writing_params.sep,
                header=writing_params.get_header(),
                index=None,
            )
            logger.msg(
                f"Recommendations successfully written in {recommendation_folder_path}"
            )
        except Exception as e:
            logger.negative(
                f"Error writing recommendations to {recommendation_folder_path}: {e}"
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
        fold_dataset: Optional[List[Dataset]],
        sep: str = "\t",
        ext: str = ".tsv",
        header: bool = True,
        column_names: List[str] | None = None,
    ) -> None:
        """This method writes the split into a local path.

        Args:
            main_dataset (Dataset): The main dataset split.
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

        # If fold data is used, we iterate over it and
        # write it locally
        if len(fold_dataset) > 0:
            for i, fold in enumerate(fold_dataset):
                fold_path = main_split_path.joinpath(str(i + 1))
                fold_path.mkdir(parents=True, exist_ok=True)
                write_dataset(fold, fold_path, "validation")

        logger.msg(f"Split data written to {main_split_path}")

    def write_time_report(self, time_report: List[Dict[str, Any]]):
        """This method writes the time report into a local path.

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
        time_report_path = join(
            self.experiment_evaluation_dir,
            f"Time_Report_{self._timestamp}{writing_params.ext}",
        )
        try:
            report = pd.DataFrame(time_report)
            float_columns = report.select_dtypes(include=["float32", "float64"]).columns

            # Format only columns relative to time
            columns_to_exclude = [
                "RAM_Mean_Usage",
                "RAM_STD_Usage",
                "RAM_Max_Usage",
                "RAM_Min_Usage",
                "VRAM_Mean_Usage",
                "VRAM_STD_Usage",
                "VRAM_Max_Usage",
                "VRAM_Min_Usage",
            ]
            columns_to_format = [
                col for col in float_columns if col not in columns_to_exclude
            ]
            for col in columns_to_format:
                report[col] = report[col].apply(format_secs)

            # Reordering columns
            first_columns = [
                "Model_Name",
                "Trainable_Params (Best Model)",
                "Total_Params (Best Model)",
            ]
            other_cols = [col for col in report.columns if col not in first_columns]
            report = report[first_columns + other_cols]

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
