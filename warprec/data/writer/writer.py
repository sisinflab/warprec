import shutil
from os.path import join
from pathlib import Path
from typing import List
from datetime import datetime
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
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
        result_dict: dict,
        model_name: str,
        top_k: List[int],
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
        result_dict: dict,
        model_name: str,
        top_k: List[int],
        validation: bool = False,
        sep: str = "\t",
        ext: str = ".tsv",
    ) -> None:
        """This function writes locally all the results of the experiment.

        Args:
            result_dict (dict): The dictionary containing the results,
                must be in the format index: value, where index
                is a string formatted as: metric_name@top_k.
            model_name (str): The name of the model which was evaluated.
            top_k (List[int]): The list of top_k, or cutoffs, to retrieve from dictionary.
            validation (bool): Flag value for the validation data.
            sep (str): The separator of the file.
            ext (str): The extension of the file.
        """
        if self.config:
            writing_params = self.config.writer.writing_params
        else:
            writing_params = WritingParams(sep=sep, ext=ext)

        _path = join(self.experiment_evaluation_dir, f"{model_name}_{self._timestamp}")
        if validation:
            _path = _path + "_Validation" + writing_params.ext
        else:
            _path = _path + "_Test" + writing_params.ext

        df = self._result_to_dataframe(result_dict, top_k)
        df.to_csv(_path, sep=writing_params.sep)

    def _result_to_dataframe(self, result_dict: dict, top_k: List[int]) -> DataFrame:
        """This is a utility method to transform a dictionary of
        results in the corresponding DataFrame format.

        Args:
            result_dict (dict): The dictionary containing the results.
                The first index is the top_k integer, the second
                the name of the metric.
            top_k (List[int]): The list of top_k, or cutoffs, to retrieve from dictionary.

        Returns:
            DataFrame: The DataFrame format of the results.
        """
        # Collect all unique metric keys across all k
        all_metric_keys = set()
        for k in top_k:
            all_metric_keys.update(result_dict[k].keys())
        sorted_metric_keys = sorted(all_metric_keys)

        result_list = []
        indexes = ["Top@" + str(k) for k in top_k]
        for k in top_k:
            row = []
            for metric in sorted_metric_keys:
                row.append(result_dict[k].get(metric, float("nan")))
            result_list.append(row)
        result_array = np.array(result_list)
        return pd.DataFrame(result_array, columns=sorted_metric_keys, index=indexes)

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

    def write_params(self, params: dict, file_name: str):
        """This method writes the model parameters into a local path.

        Args:
            params (dict): The parameters of the model.
            file_name (str): The name used to save the parameters.
        """
        # experiment_path/serialized/model_name_params.json
        _path = join(
            self.experiment_serialized_models_dir,
            f"{file_name}_params_{self._timestamp}.json",
        )
        with open(_path, "w") as f:
            json.dump(params, f, indent=4)

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

    def write_overall_results(
        self, overall_results: dict, sep: str = "\t", ext: str = ".tsv"
    ):
        """This method writes the overall results of the experiment.

        Args:
            overall_results (dict): The dictionary containing the overall results.
                The first index is the model name, the second is the set of results,
                which can be either "Validation" or "Test", the third is the cutoff
                and the last is the metric name.
            sep (str): The separator that will be used to write the results.
            ext (str): The extension that will be used to write the results.
        """
        if self.config:
            writing_params = self.config.writer.writing_params
        else:
            writing_params = WritingParams(sep=sep, ext=ext)

        # experiment_path/overall_results.{custom_extension}
        _path = join(
            self.experiment_evaluation_dir,
            f"overall_results_{self._timestamp}" + writing_params.ext,
        )

        # Convert overall results to DataFrame
        result_list = []
        for model_name, result_dict in overall_results.items():
            for set_name, metrics in result_dict.items():
                for top_k, metric_values in metrics.items():
                    row = {"Model": model_name, "Set": set_name, "Top@k": top_k}
                    row.update(metric_values)
                    result_list.append(row)

        df = pd.DataFrame(result_list)
        df.to_csv(_path, sep=writing_params.sep, index=False)

    def checkpoint_from_ray(self, source: str, new_name: str):
        destination = join(self.experiment_serialized_models_dir, new_name + ".pth")
        shutil.move(source, destination)
