from typing import Any, Tuple
from abc import ABC, abstractmethod

from pandas import DataFrame
from warprec.utils.config import (
    WarpRecConfiguration,
)


class Reader(ABC):
    """The abstract definition of a reader. All readers must extend this class.

    Attributes:
        config (WarpRecConfiguration): Configuration file.

    TODO: Use Factory Pattern for different reader.
    """

    config: WarpRecConfiguration = None

    @abstractmethod
    def read(self, **kwargs: Any) -> DataFrame:
        """This method will read the data from the source."""

    @abstractmethod
    def load_model_state(self, **kwargs: Any) -> dict:
        """This method will load a model state from a source."""

    @abstractmethod
    def read_transaction_split(
        self, **kwargs: Any
    ) -> Tuple[DataFrame, DataFrame | None, DataFrame | None]:
        """This method will read the split data from the source."""
