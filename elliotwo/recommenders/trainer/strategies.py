# pylint: disable=unused-argument, too-few-public-methods, too-many-arguments, too-many-positional-arguments
from abc import ABC

from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import (
    FIFOScheduler,
    ASHAScheduler,
)
from elliotwo.utils.registry import search_algorithm_registry, scheduler_registry


@search_algorithm_registry.register("grid")
class GridSearchWrapper(ABC):
    """Wrapper for the GridSearch algorithm in Ray Tune.

    This wrapper is empty in order to be registered inside
    the search_algorithm_registry but return None to Ray Tune.
    """

    def __init__(self, **kwargs):
        pass

    def __new__(cls):
        return None


@search_algorithm_registry.register("random")
class RandomSearchWrapper(ABC):
    """Wrapper for the RandomSearch algorithm in Ray Tune.

    This wrapper is empty in order to be registered inside
    the search_algorithm_registry but return None to Ray Tune.
    """

    def __init__(self, **kwargs):
        pass

    def __new__(cls):
        return None


@search_algorithm_registry.register("hopt")
class HyperOptWrapper(HyperOptSearch, ABC):
    """Wrapper for the HyperOpt algorithm in Ray Tune.

    Args:
        mode (str): The mode to run the optimization. Must be
            either 'min' or 'max'.
        seed (int): The seed to make the experiment reproducible.
        **kwargs: Keyword arguments.
    """

    def __init__(self, mode: str, seed: int, **kwargs):
        super().__init__(mode=mode, random_state_seed=seed, metric="score")


@search_algorithm_registry.register("optuna")
class OptunaWrapper(OptunaSearch, ABC):
    """Wrapper for the HyperOpt algorithm in Ray Tune.

    Args:
        mode (str): The mode to run the optimization. Must be
            either 'min' or 'max'.
        seed (int): The seed to make the experiment reproducible.
        **kwargs: Keyword arguments.
    """

    def __init__(self, mode: str, seed: int, **kwargs):
        super().__init__(mode=mode, seed=seed, metric="score")


@search_algorithm_registry.register("bohb")
class BOHBWrapper(TuneBOHB, ABC):
    """Wrapper for the HyperOpt algorithm in Ray Tune.

    Args:
        mode (str): The mode to run the optimization. Must be
            either 'min' or 'max'.
        seed (int): The seed to make the experiment reproducible.
        **kwargs: Keyword arguments.
    """

    def __init__(self, mode: str, seed: int, **kwargs):
        super().__init__(mode=mode, seed=seed, metric="score")


@scheduler_registry.register("fifo")
class FIFOSchedulerWrapper(FIFOScheduler, ABC):
    """Wrapper for the FIFO scheduler."""

    def __init__(self, **kwargs):
        pass


@scheduler_registry.register("asha")
class ASHASchedulerWrapper(ASHAScheduler, ABC):
    """Wrapper for the ASHA scheduler.

    Args:
        mode (str): The mode to run the optimization. Must be
            either 'min' or 'max'.
        time_attr (str): The measure of time that will be used
            by the scheduler.
        max_t (int): Max time unit given to each trial.
        grace_period (int): Min time unit given to each trial.
        reduction_factor (float): Halving rate of trials.
        **kwargs: Keyword arguments.
    """

    def __init__(
        self,
        mode: str,
        time_attr: str,
        max_t: int,
        grace_period: int,
        reduction_factor: float,
        **kwargs,
    ):
        super().__init__(
            mode=mode,
            time_attr=time_attr,
            max_t=max_t,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
            metric="score",
        )
