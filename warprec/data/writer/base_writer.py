from typing import Dict, List, Optional
from abc import ABC, abstractmethod


from pandas import DataFrame
from torch import Tensor

from warprec.utils.config import TrainConfiguration
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import (
    Recommender,
)


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
    def write_recs(
        self,
        model: Recommender,
        dataset: Dataset,
        k: int,
    ):
        """This method writes recommendations in the destination."""

    @abstractmethod
    def write_model(self, model: Recommender):
        """This method writes the model state into a destination."""

    @abstractmethod
    def write_split(
        self,
        main_dataset: Dataset,
        val_dataset: Optional[DataFrame],
        fold_dataset: Optional[List[Dataset]],
    ):
        """This method writes the split of the dataset into a destination."""
