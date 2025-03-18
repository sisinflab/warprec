# pylint: disable=arguments-differ
from abc import abstractmethod, ABC
from typing import Any

import torch
from torch import Tensor
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

    def dcg(self, rel: Tensor) -> Tensor:
        """The Discounted Cumulative Gain definition.

        Args:
            rel (Tensor): The relevance tensor.

        Returns:
            Tensor: The discounted tensor.
        """
        return (
            rel / torch.log2(torch.arange(2, rel.size(1) + 2, device=rel.device))
        ).sum(dim=1)

    def discounted_sum(self, k: int) -> Tensor:
        """"""
        ranks = torch.arange(k)
        return torch.sum(1.0 / torch.log2(ranks.float() + 2))
