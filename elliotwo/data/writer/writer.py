import time
from os.path import join
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import torch
from pandas import DataFrame
from elliotwo.utils.config import Configuration
from elliotwo.data.dataset import AbstractDataset
from elliotwo.recommenders.abstract_recommender import AbstractRecommender
from elliotwo.utils.logger import logger


class AbstractWriter(ABC):
    """AbstractWriter is the abstract definition of a writer,
    during an experiment multiple writers can be defined.

    Args:
        config (Configuration): The configuration of the experiment.

    TODO: Use Factory Pattern for different writer.
    """

    def __init__(self, config: Configuration):
        self._config = config
        self._experiment_name = self._config.data.dataset_name

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
        metric_names: List[str],
        top_k: List[int],
    ):
        """This function writes all the results of the experiment."""

    @abstractmethod
    def write_recs(self, recs: DataFrame, model_name: str):
        """This method writes recommendations in the destination."""

    @abstractmethod
    def write_model(self, model: AbstractRecommender):
        """This method writes the model state into a destination."""

    @abstractmethod
    def write_split(self, dataset: AbstractDataset):
        """This method writes the split of the dataset into a destination."""


class LocalWriter(AbstractWriter):
    """LocalWriter is the class to be used when the results of
    the experiment want to be saved locally.

    Args:
        config (Configuration): The configuration of the experiment.

    TODO: Using context manager
    """

    def __init__(self, config: Configuration):
        super().__init__(config)
        self._timestamp = str(int(time.time() * 1000))
        self._experiment_path = Path(
            join(
                self._config.data.experiment_path,
                self._experiment_name,
                self._timestamp,
            )
        )
        self._experiment_evaluation_dir = Path(
            join(self._experiment_path, "evaluation")
        )
        self._experiment_recommendation_dir = Path(join(self._experiment_path, "recs"))
        self._experiment_serialized_models_dir = Path(
            join(self._experiment_path, "serialized")
        )
        self._experiment_split_dir = Path(join(self._experiment_path, "split"))

        if self._config.general.setup_experiment:
            self.setup_experiment()

    def setup_experiment(self):
        """This is the main function to be executed, it sets up all
        the important directory to then later save results."""
        logger.msg("Setting up experiment local folder.")

        # Check if directory exists and create the non existing one
        self._experiment_path.mkdir(parents=True, exist_ok=True)
        self._experiment_evaluation_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_recommendation_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_serialized_models_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_split_dir.mkdir(parents=True, exist_ok=True)

        # Write locally the json file of the configuration
        json_dump = self._config.model_dump_json(indent=2)
        json_path = Path(join(self._experiment_path, "config.json"))
        json_path.write_text(json_dump, encoding="utf-8")

        logger.msg("Experiment folder created successfully.")

    def write_results(
        self,
        result_dict: dict,
        model_name: str,
        metric_names: List[str],
        top_k: List[int],
    ) -> None:
        """This function writes locally all the results of the experiment.

        Args:
            result_dict (dict): The dictionary containing the results,
                must be in the format index: value, where index
                is a string formatted as: metric_name@top_k.
            model_name (str): The name of the model which was evaluated.
            metric_names (List[str]): The names of the metrics to be retrieved from the dictionary.
            top_k (List[int]): The list of top_k, or cutoffs, to retrieve from dictionary.
        """
        _path = join(self._experiment_evaluation_dir, model_name)
        if self._config.splitter.validation:
            _val_path = _path + "_Validation" + self._config.general.recommendation.ext
            df = self._result_to_dataframe(
                result_dict["Validation"], metric_names, top_k
            )
            df.to_csv(_val_path, sep=self._config.general.recommendation.sep)
        _test_path = _path + "_Test" + self._config.general.recommendation.ext
        df = self._result_to_dataframe(result_dict["Test"], metric_names, top_k)
        df.to_csv(_test_path, sep=self._config.general.recommendation.sep)

    def _result_to_dataframe(
        self, result_dict: dict, metric_names: List[str], top_k: List[int]
    ) -> DataFrame:
        """This is a utility method to transform a dictionary of
        results in the corresponding DataFrame format.

        Args:
            result_dict (dict): The dictionary containing the results.
                The first index is the top_k integer, the second
                the name of the metric.
            metric_names (List[str]): The names of the metrics to be retrieved from the dictionary.
            top_k (List[int]): The list of top_k, or cutoffs, to retrieve from dictionary.

        Returns:
            DataFrame: The DataFrame format of the results.
        """
        result_list = []
        indexes = ["Top@" + str(k) for k in top_k]
        for k in top_k:
            row = []
            for metric in metric_names:
                row.append(result_dict[k][metric])
            result_list.append(row)
        result_array = np.array(result_list)
        return pd.DataFrame(result_array, columns=metric_names, index=indexes)

    def write_recs(self, recs: DataFrame, model_name: str) -> None:
        """This method writes recommendations in the local path.

        Args:
            recs (DataFrame): The recommendations in DataFrame format.
            model_name (str): The name of the model which produced the recommendations.
        """
        # experiment_path/recs/model_name.{custom_extension}
        _path = join(
            self._experiment_recommendation_dir,
            model_name + self._config.general.recommendation.ext,
        )

        # Save in path
        recs.to_csv(
            _path,
            sep=self._config.general.recommendation.sep,
            header=[
                self._config.data.labels.user_id_label,
                self._config.data.labels.item_id_label,
            ],
            index=None,
        )

    def write_model(self, model: AbstractRecommender):
        """This method writes the model state into a local path,
        using joblib for the serialization.

        Args:
            model (AbstractRecommender): The model to write locally.
        """
        # experiment_path/serialized/model_name.joblib
        _path = join(self._experiment_serialized_models_dir, model.name_param + ".pth")
        torch.save(model.state_dict(), _path)

    def write_split(self, dataset: AbstractDataset) -> None:
        _path_train = join(self._experiment_split_dir, "train.tsv")
        _path_test = join(self._experiment_split_dir, "test.tsv")
        _path_val = join(self._experiment_split_dir, "val.tsv")

        if dataset.train_set is not None:
            dataset.train_set.get_df().to_csv(_path_train, sep="\t", index=None)
        if dataset.test_set is not None:
            dataset.test_set.get_df().to_csv(_path_test, sep="\t", index=None)
        if dataset.val_set is not None:
            dataset.val_set.get_df().to_csv(_path_val, sep="\t", index=None)
