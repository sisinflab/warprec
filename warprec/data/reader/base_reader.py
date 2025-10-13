from typing import Any, Tuple
from abc import ABC, abstractmethod

from pandas import DataFrame

from warprec.utils.config import (
    WarpRecConfiguration,
)
from warprec.utils.enums import ReadingMethods


class Reader(ABC):
    """The abstract definition of a reader. All readers must extend this class.

    Attributes:
        config (WarpRecConfiguration): Configuration file.
    """

    config: WarpRecConfiguration = None

    @abstractmethod
    def read(self, *args: Any, **kwargs: Any) -> DataFrame:
        """This method will read the data from the source."""

    @abstractmethod
    def load_model_state(self, *args: Any, **kwargs: Any) -> dict:
        """This method will load a model state from a source."""

    @abstractmethod
    def read_transaction_split(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[DataFrame, DataFrame | None, DataFrame | None]:
        """This method will read the split data from the source."""

    @abstractmethod
    def read_side_information(self, *args: Any, **kwargs: Any) -> DataFrame:
        """This method will read the side information from a source."""

    @abstractmethod
    def read_cluster_information(
        self, *args: Any, **kwargs: Any
    ) -> Tuple[DataFrame, DataFrame]:
        """This method will read the cluster information (user and item) from a source."""


class ReaderFactory:  # pylint: disable=C0415, R0903
    """Factory class for creating Reader instances based on configuration.

    Attributes:
        config (WarpRecConfiguration): Configuration file.
    """

    config: WarpRecConfiguration = None

    @classmethod
    def get_reader(cls, config: WarpRecConfiguration) -> Reader:
        """Factory method to get the appropriate Reader instance based on the configuration.

        Args:
            config (WarpRecConfiguration): Configuration file.

        Returns:
            Reader: An instance of a class that extends the Reader abstract class.

        Raises:
            ValueError: If the reading method specified in the configuration is unknown.
        """
        reader_type = config.reader.reading_method

        # Create the appropriate Reader instance based on the reading method
        match reader_type:
            case ReadingMethods.LOCAL:
                from warprec.data.reader import LocalReader

                return LocalReader(config=config)
            case ReadingMethods.AZURE_BLOB:
                from warprec.data.reader import AzureBlobReader

                return AzureBlobReader(config=config)

        raise ValueError(f"Unknown reader type: {reader_type}")
