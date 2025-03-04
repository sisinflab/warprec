# pylint: disable=unused-argument, too-few-public-methods
"""
This script contains the wrapper of the Ray wrappers for the schedulers.
At the time of writing this there is no common interface provided by Ray.
This makes the process of registering these classes not very 'pythonic',
but it serves its purpose. In future this class must be refactored if possible.

TODO: Refactor this script in a more pythonic way.

Author: Avolio Marco
Date: 03/03/2025
"""

from typing import Any
from abc import ABC, abstractmethod

from ray.tune.schedulers import (
    FIFOScheduler,
    ASHAScheduler,
)
from elliotwo.utils.enums import Schedulers
from elliotwo.utils.registry import scheduler_registry


class BaseSchedulerWrapper(ABC):
    """Common interface for all scheduler wrappers."""

    @abstractmethod
    def __init__(self, **kwargs: Any):
        pass


@scheduler_registry.register(Schedulers.FIFO)
class FIFOSchedulerWrapper(FIFOScheduler, BaseSchedulerWrapper):
    """Wrapper for the FIFO scheduler.

    Args:
        **kwargs (Any): Keyword arguments.
    """

    def __init__(self, **kwargs: Any):
        return None


@scheduler_registry.register(Schedulers.ASHA)
class ASHASchedulerWrapper(ASHAScheduler, BaseSchedulerWrapper):
    """Wrapper for the ASHA scheduler.

    Args:
        mode (str): The mode to run the optimization. Must be
            either 'min' or 'max'.
        time_attr (str): The measure of time that will be used
            by the scheduler.
        max_t (int): Maximum number of iterations.
        grace_period (int): Min time unit given to each trial.
        reduction_factor (float): Halving rate of trials.
        **kwargs (Any): Keyword arguments.
    """

    def __init__(
        self,
        mode: str,
        time_attr: str,
        max_t: int,
        grace_period: int,
        reduction_factor: float,
        **kwargs: Any,
    ):
        super().__init__(
            mode=mode,
            time_attr=time_attr,
            max_t=max_t,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
            metric="score",
        )
