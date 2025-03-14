# pylint: disable=arguments-differ
from abc import abstractmethod, ABC
from typing import Any

from torchmetrics import Metric


class BaseMetric(Metric, ABC):
    """The base definition of a metric using Torchmetrics."""

    @abstractmethod
    def compute(self):
        pass

    @property
    def name(self):
        """The name of the metric."""
        return self.__class__.__name__


class TopKMetric(BaseMetric):
    """The definition of a Top-K metric."""

    def __init__(self, k: int, dist_sync_on_step=False, *args: Any, **kwargs: Any):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = k
