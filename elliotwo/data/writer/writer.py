import time
import shutil
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
from elliotwo.utils.enums import WritingMethods
from elliotwo.utils.config import WriterConfig, WritingResultConfig
from elliotwo.utils.logger import logger


class AbstractWriter(ABC):
    """AbstractWriter is the abstract definition of a writer,
    during an experiment multiple writers can be defined.

    Attributes:
        config (Configuration | None): The configuration of the experiment.

    TODO: Use Factory Pattern for different writer.
    """

    config: Configuration | None

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

    @abstractmethod
    def checkpoint_from_ray(self, source: str, new_name: str):
        """This method takes a ray checkpoint and moves to a destination."""


class LocalWriter(AbstractWriter):
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

            # Setup experiment information from config
            self.experiment_name = config.writer.dataset_name
            self.local_path = config.writer.local_experiment_path
            self.setup = config.writer.setup_experiment
        else:
            # Setup experiment information from args
            self.experiment_name = dataset_name
            self.local_path = local_path
            self.setup = setup

        writer_params = WriterConfig(
            dataset_name=self.experiment_name,
            writing_method=WritingMethods.LOCAL,
            local_experiment_path=self.local_path,
            setup_experiment=self.setup,
        )

        self._timestamp = str(int(time.time() * 1000))
        self.experiment_path = Path(
            join(
                writer_params.local_experiment_path,
                writer_params.dataset_name,
                self._timestamp,
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
            json_path = Path(join(self.experiment_path, "config.json"))
            json_path.write_text(json_dump, encoding="utf-8")

        logger.msg("Experiment folder created successfully.")

    def write_results(
        self,
        result_dict: dict,
        model_name: str,
        metric_names: List[str],
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
            metric_names (List[str]): The names of the metrics to be retrieved from the dictionary.
            top_k (List[int]): The list of top_k, or cutoffs, to retrieve from dictionary.
            validation (bool): Flag value for the validation data.
            sep (str): The separator of the file.
            ext (str): The extension of the file.
        """
        if self.config:
            _sep = self.config.writer.result.sep
            _ext = self.config.writer.result.ext
        else:
            _sep = sep
            _ext = ext

        result_params = WritingResultConfig(sep=_sep, ext=_ext)

        _path = join(self.experiment_evaluation_dir, model_name)
        if validation:
            _path = _path + "_Validation" + result_params.ext
        else:
            _path = _path + "_Test" + result_params.ext

        df = self._result_to_dataframe(result_dict, metric_names, top_k)
        df.to_csv(_path, sep=result_params.sep)

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

    def write_recs(
        self,
        recs: DataFrame,
        model_name: str,
        sep: str = "\t",
        ext: str = ".tsv",
        user_label: str = "user_id",
        item_label: str = "item_id",
    ) -> None:
        """This method writes recommendations in the local path.

        Args:
            recs (DataFrame): The recommendations in DataFrame format.
            model_name (str): The name of the model which produced the recommendations.
            sep (str): The separator of the file.
            ext (str): The extension of the file.
            user_label (str): The label of the user data.
            item_label (str): The label of the item data.
        """
        if self.config:
            _sep = self.config.writer.result.sep
            _ext = self.config.writer.result.ext
            _user_label = self.config.writer.result.user_label
            _item_label = self.config.writer.result.item_label
        else:
            _sep = sep
            _ext = ext
            _user_label = user_label
            _item_label = item_label

        result_params = WritingResultConfig(
            sep=_sep, ext=_ext, user_label=_user_label, item_label=_item_label
        )

        # experiment_path/recs/model_name.{custom_extension}
        _path = join(
            self.experiment_recommendation_dir,
            model_name + result_params.ext,
        )

        # Save in path
        recs.to_csv(
            _path,
            sep=result_params.sep,
            header=[
                result_params.user_label,
                result_params.item_label,
            ],
            index=None,
        )

    def write_model(self, model: AbstractRecommender):
        """This method writes the model state into a local path.

        Args:
            model (AbstractRecommender): The model to write locally.
        """
        # experiment_path/serialized/model_name.pth
        _path = join(self.experiment_serialized_models_dir, model.name_param + ".pth")
        torch.save(model.state_dict(), _path)

    def write_split(
        self, dataset: AbstractDataset, sep: str = "\t", ext: str = ".tsv"
    ) -> None:
        """This method writes the split into a local path.

        Args:
            dataset (AbstractDataset): The dataset splitted.
            sep (str): The separator that will be used to write the results.
            ext (str): The extension that will be used to write the results.
        """
        if self.config:
            _sep = self.config.writer.result.sep
            _ext = self.config.writer.result.ext
        else:
            _sep = sep
            _ext = ext

        result_params = WritingResultConfig(sep=_sep, ext=_ext)

        path_train = join(self.experiment_split_dir, "train" + result_params.ext)
        path_test = join(self.experiment_split_dir, "test" + result_params.ext)
        path_val = join(self.experiment_split_dir, "val" + result_params.ext)

        if dataset.train_set is not None:
            dataset.train_set.get_df().to_csv(
                path_train, sep=result_params.sep, index=None
            )
        if dataset.test_set is not None:
            dataset.test_set.get_df().to_csv(
                path_test, sep=result_params.sep, index=None
            )
        if dataset.val_set is not None:
            dataset.val_set.get_df().to_csv(path_val, sep=result_params.sep, index=None)

    def checkpoint_from_ray(self, source: str, new_name: str):
        destination = join(self.experiment_serialized_models_dir, new_name + ".pth")
        shutil.move(source, destination)
