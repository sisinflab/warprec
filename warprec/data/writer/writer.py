import shutil
from os.path import join
from pathlib import Path
from typing import Dict
from datetime import datetime
from abc import ABC, abstractmethod

import pandas as pd
import torch
import json
from pandas import DataFrame
from warprec.utils.config import Configuration
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.enums import WritingMethods
from warprec.utils.config import WriterConfig, WritingParams
from warprec.utils.logger import logger


class Writer(ABC):
    """Writer is the abstract definition of a writer,
    during an experiment multiple writers can be defined.

    Attributes:
        config (Configuration): The configuration of the experiment.

    TODO: Use Factory Pattern for different writer.
    """

    config: Configuration = None

    @abstractmethod
    def setup_experiment(self):
        """This is the main function to be executed, it sets up all
        the important directory to then later save results.
        """

    @abstractmethod
    def write_results(
        self,
        result_dict: Dict[str, Dict[int, Dict[str, float]]],
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
    def write_split(self, dataset: Dataset):
        """This method writes the split of the dataset into a destination."""

    @abstractmethod
    def checkpoint_from_ray(self, source: str, new_name: str):
        """This method takes a ray checkpoint and moves to a destination."""


class LocalWriter(Writer):
    """LocalWriter is the class to be used when the results of
    the experiment want to be saved locally.

    Args:
        dataset_name (str): The name of the dataset.
        local_path (str): The path to the dataset.
        config (Configuration): The configuration of the experiment.
        setup (bool): Flag value for the setup of the experiment.

    TODO: Using context manager
    """

    def __init__(
        self,
        dataset_name: str = None,
        local_path: str = None,
        config: Configuration = None,
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
        self.experiment_split_dir = Path(join(self.experiment_path, "split"))

        if writer_params.setup_experiment:
            self.setup_experiment(config)

    def setup_experiment(self, config: Configuration = None):
        """This is the main function to be executed, it sets up all
        the important directory to then later save results."""
        logger.msg("Setting up experiment local folder.")

        # Check if directory exists and create the non existing one
        self.experiment_path.mkdir(parents=True, exist_ok=True)
        self.experiment_evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_recommendation_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_serialized_models_dir.mkdir(parents=True, exist_ok=True)
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
        result_data: Dict[str, Dict[int, Dict[str, float]]],
        model_name: str,
        sep: str = "\t",
        ext: str = ".tsv",
    ) -> None:
        """This function writes locally all the results of the experiment into a single
        "Overall_Results_{timestamp}.tsv" file, merging with existing data if present.

        Args:
            result_data (Dict[str, Dict[int, Dict[str, float]]]): The dictionary containing the results.
                Format: { "Set": { "k": { "MetricName": value } } }
                Example: {"Test": {5: {"Precision": 0.1, "Recall": 0.2}}}
            model_name (str): The name of the model which was evaluated.
            sep (str): The separator of the file.
            ext (str): The extension of the file.
        """
        if self.config:
            writing_params = self.config.writer.writing_params
        else:
            writing_params = WritingParams(sep=sep, ext=ext)

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
        for set_name, top_k_data in result_data.items():
            for k_value, metrics in top_k_data.items():
                row = {"Model": model_name, "Set": set_name, "Top@k": k_value}
                row.update(metrics)
                new_result_list.append(row)

        new_df = pd.DataFrame(new_result_list)

        # Merge new results with existing ones
        if not existing_df.empty:
            # We need to ensure columns match before concatenation.
            # If there are new metrics, add them to existing_df with NaN.
            # If existing_df has metrics not in new_df, add them to new_df with NaN.
            all_columns = list(set(existing_df.columns) | set(new_df.columns))
            existing_df = existing_df.reindex(columns=all_columns)
            new_df = new_df.reindex(columns=all_columns)

            # Define merge keys for deduplication
            merge_keys = ["Model", "Set", "Top@k"]

            # Filter out rows from existing_df that are exactly matched by new_df based on merge_keys
            # This handles updates for existing model/set/k combinations.
            # First, concatenate to get all data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Drop duplicates, keeping the 'last' (i.e., the one from new_df if it's a duplicate)
            # This means if a model/set/k combination is present in both, the new one takes precedence.
            final_df = combined_df.drop_duplicates(subset=merge_keys, keep="last")
        else:
            final_df = new_df

        # Sort for consistent output (optional but good practice)
        final_df = final_df.sort_values(by=["Model", "Set", "Top@k"]).reset_index(
            drop=True
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
            user_label (str): The label of the user data.
            item_label (str): The label of the item data.
            rating_label (str): The label of the rating data.
        """
        if self.config:
            writing_params = self.config.writer.writing_params
        else:
            writing_params = WritingParams(
                sep=sep,
                ext=ext,
                user_label=user_label,
                item_label=item_label,
                rating_label=rating_label,
            )

        # experiment_path/recs/model_name.{custom_extension}
        _path = join(
            self.experiment_recommendation_dir,
            f"{model_name}_{self._timestamp}{writing_params.ext}",
        )

        # Save in path
        recs.to_csv(
            _path,
            sep=writing_params.sep,
            header=[
                writing_params.user_label,
                writing_params.item_label,
                writing_params.rating_label,
            ],
            index=None,
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
        _path = (
            self.experiment_serialized_models_dir
            / f"Overall_Params_{self._timestamp}.json"
        )

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

    def write_split(self, dataset: Dataset, sep: str = "\t", ext: str = ".tsv") -> None:
        """This method writes the split into a local path.

        Args:
            dataset (Dataset): The dataset splitted.
            sep (str): The separator that will be used to write the results.
            ext (str): The extension that will be used to write the results.
        """
        if self.config:
            writing_params = self.config.writer.writing_params
        else:
            writing_params = WritingParams(sep=sep, ext=ext)

        path_train = join(self.experiment_split_dir, "train" + writing_params.ext)
        path_test = join(self.experiment_split_dir, "test" + writing_params.ext)
        path_val = join(self.experiment_split_dir, "val" + writing_params.ext)

        if dataset.train_set is not None:
            dataset.train_set.get_df().to_csv(
                path_train, sep=writing_params.sep, index=None
            )
        if dataset.test_set is not None:
            dataset.test_set.get_df().to_csv(
                path_test, sep=writing_params.sep, index=None
            )
        if dataset.val_set is not None:
            dataset.val_set.get_df().to_csv(
                path_val, sep=writing_params.sep, index=None
            )

    def checkpoint_from_ray(self, source: str, new_name: str):
        destination = join(self.experiment_serialized_models_dir, new_name + ".pth")
        shutil.move(source, destination)
