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

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from ray.tune.schedulers import (
    FIFOScheduler,
    ASHAScheduler,
    HyperBandForBOHB,
    MedianStoppingRule,
)
from warprec.utils.enums import Schedulers
from warprec.utils.registry import scheduler_registry


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

    def __init__(self, **kwargs: Any):  # pylint: disable=W0231
        return None


@scheduler_registry.register(Schedulers.ASHA)
class ASHASchedulerWrapper(ASHAScheduler, BaseSchedulerWrapper):
    """Wrapper for the ASHA scheduler.

    Args:
        max_t (int): Maximum number of iterations.
        grace_period (int): Min time unit given to each trial.
        reduction_factor (float): Halving rate of trials.
        time_attr (Optional[str]): The measure of time that will be used
            by the scheduler. Defaults to 'training_iteration' when not given.
        **kwargs (Any): Keyword arguments.

    Note:
        A time attribute must always reach Ray Tune. ASHA compares it against
        the reported results and, when it does not appear among them, lets every
        trial continue: a scheduler configured without one would silently stop
        pruning and behave like FIFO.
    """

    def __init__(
        self,
        max_t: int,
        grace_period: int,
        reduction_factor: float,
        time_attr: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            time_attr=time_attr or "training_iteration",
            max_t=max_t,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
        )


@scheduler_registry.register(Schedulers.BOHB)
class BOHBSchedulerWrapper(HyperBandForBOHB, BaseSchedulerWrapper):
    """Wrapper for the BOHB scheduler.

    This scheduler is the HyperBand half of BOHB and must be paired with the
    'bohb' search algorithm: it pauses trials, and only the BOHB search
    algorithm knows how to resume them and insert new ones in their place.

    Unlike ASHA, this scheduler has no grace period.

    Args:
        max_t (int): Maximum number of iterations.
        reduction_factor (float): Halving rate of trials.
        time_attr (Optional[str]): The measure of time that will be used
            by the scheduler. Defaults to 'training_iteration' when not given.
        **kwargs (Any): Keyword arguments.
    """

    def __init__(
        self,
        max_t: int,
        reduction_factor: float,
        time_attr: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            time_attr=time_attr or "training_iteration",
            max_t=max_t,
            reduction_factor=reduction_factor,
        )


@scheduler_registry.register(Schedulers.MEDIAN)
class MedianStoppingRuleWrapper(MedianStoppingRule, BaseSchedulerWrapper):
    """Wrapper for the median stopping rule scheduler.

    The scheduler stops a trial whose performance, at a given point in time,
    falls below the median of the trials observed so far.

    Args:
        grace_period (float): How old a trial must be before it can be stopped.
            The unit is the one named by 'time_attr'.
        time_attr (Optional[str]): The measure of time that will be used
            by the scheduler. Defaults to 'training_iteration' when not given.
        min_samples_required (Optional[int]): Minimum number of trials to
            compute the median over. Ray's default is kept when not given.
        min_time_slice (Optional[int]): How long a trial runs before yielding.
            The unit is the one named by 'time_attr'. Ray's default is kept
            when not given.
        hard_stop (Optional[bool]): Whether to stop trials outright. When False,
            trials are paused instead and resumed FIFO once the others have
            finished. Ray's default is kept when not given.
        **kwargs (Any): Keyword arguments.

    Note:
        Ray defaults 'time_attr' to 'time_total_s' for this scheduler, but
        WarpRec defaults it to 'training_iteration' as it does for the other
        schedulers. Since 'grace_period' and 'min_time_slice' are expressed in
        the units of 'time_attr', keeping one default across schedulers stops
        the same configuration value from meaning seconds under one scheduler
        and iterations under another.
    """

    def __init__(
        self,
        grace_period: float,
        time_attr: Optional[str] = None,
        min_samples_required: Optional[int] = None,
        min_time_slice: Optional[int] = None,
        hard_stop: Optional[bool] = None,
        **kwargs: Any,
    ):
        # Only forward what the user actually set, so that the defaults of the
        # underlying Ray scheduler are not duplicated here. The annotation is
        # needed because these parameters do not share a single type.
        optional_params: Dict[str, Any] = {
            "min_samples_required": min_samples_required,
            "min_time_slice": min_time_slice,
            "hard_stop": hard_stop,
        }
        super().__init__(
            time_attr=time_attr or "training_iteration",
            grace_period=grace_period,
            **{k: v for k, v in optional_params.items() if v is not None},
        )
