from typing import Dict, List, Optional
from abc import ABC, abstractmethod


from pandas import DataFrame
from torch import Tensor

from warprec.data.dataset import Dataset
from warprec.data.writer import LocalWriter, AzureBlobWriter
from warprec.recommenders.base_recommender import (
    Recommender,
)
from warprec.utils.config import TrainConfiguration
from warprec.utils.enums import WritingMethods


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


class WriterFactory:
    """Factory class for creating Writer instances based on configuration.

    Attributes:
        config (TrainConfiguration): The configuration of the experiment.
    """

    config: TrainConfiguration = None

    @classmethod
    def get_writer(cls, config: TrainConfiguration) -> Writer:
        """Factory method to get the appropriate Writer instance based on the configuration.

        Args:
            config (TrainConfiguration): The configuration of the experiment.

        Returns:
            Writer: An instance of a class that extends the Writer abstract class.

        Raises:
            ValueError: If the writing method specified in the configuration is unknown.
        """
        writer_type = config.writer.writing_method

        # Create the appropriate Writer instance based on the writing method
        if writer_type == WritingMethods.LOCAL:
            return LocalWriter(config=config)
        elif writer_type == WritingMethods.AZURE_BLOB:
            return AzureBlobWriter(config=config)
        else:
            raise ValueError(f"Unknown writer type: {writer_type}")
