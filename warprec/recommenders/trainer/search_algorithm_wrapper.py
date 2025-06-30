# pylint: disable=unused-argument, too-few-public-methods
"""
This script contains the wrapper of the Ray wrappers for the search algorithms.
At the time of writing this there is no common interface provided by Ray.
This makes the process of registering these classes not very 'pythonic',
but it serves its purpose. In future this class must be refactored if possible.

TODO: Refactor this script in a more pythonic way.

Author: Avolio Marco
Date: 03/03/2025
"""

from typing import Any
from abc import ABC, abstractmethod

from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bohb import TuneBOHB
from warprec.utils.enums import SearchAlgorithms
from warprec.utils.registry import search_algorithm_registry


class BaseSearchWrapper(ABC):
    """Common interface for all search algorithm wrappers."""

    @abstractmethod
    def __init__(self, **kwargs: Any):
        pass


@search_algorithm_registry.register(SearchAlgorithms.GRID)
class GridSearchWrapper(BaseSearchWrapper):
    """Wrapper for the GridSearch algorithm in Ray Tune.

    This wrapper is empty in order to be registered inside
    the search_algorithm_registry but return None to Ray Tune.

    Args:
        **kwargs (Any): Keyword arguments.
    """

    def __init__(self, **kwargs: Any):
        pass

    def __new__(cls, **kwargs: Any):
        return None


@search_algorithm_registry.register(SearchAlgorithms.RANDOM)
class RandomSearchWrapper(BaseSearchWrapper):
    """Wrapper for the RandomSearch algorithm in Ray Tune.

    This wrapper is empty in order to be registered inside
    the search_algorithm_registry but return None to Ray Tune.

    Args:
        **kwargs (Any): Keyword arguments.
    """

    def __init__(self, **kwargs: Any):
        pass

    def __new__(cls, **kwargs: Any):
        return None


@search_algorithm_registry.register(SearchAlgorithms.HYPEROPT)
class HyperOptWrapper(HyperOptSearch, BaseSearchWrapper):
    """Wrapper for the HyperOpt algorithm in Ray Tune.

    Args:
        mode (str): The mode to run the optimization. Must be
            either 'min' or 'max'.
        seed (int): The seed to make the experiment reproducible.
        **kwargs (Any): Keyword arguments.
    """

    def __init__(self, mode: str, seed: int, **kwargs: Any):
        super().__init__(mode=mode, random_state_seed=seed, metric="score")


@search_algorithm_registry.register(SearchAlgorithms.OPTUNA)
class OptunaWrapper(OptunaSearch, BaseSearchWrapper):
    """Wrapper for the HyperOpt algorithm in Ray Tune.

    Args:
        mode (str): The mode to run the optimization. Must be
            either 'min' or 'max'.
        seed (int): The seed to make the experiment reproducible.
        **kwargs (Any): Keyword arguments.
    """

    def __init__(self, mode: str, seed: int, **kwargs: Any):
        super().__init__(mode=mode, seed=seed, metric="score")


@search_algorithm_registry.register(SearchAlgorithms.BOHB)
class BOHBWrapper(TuneBOHB, BaseSearchWrapper):
    """Wrapper for the HyperOpt algorithm in Ray Tune.

    Args:
        mode (str): The mode to run the optimization. Must be
            either 'min' or 'max'.
        seed (int): The seed to make the experiment reproducible.
        **kwargs (Any): Keyword arguments.
    """

    def __init__(self, mode: str, seed: int, **kwargs: Any):
        super().__init__(mode=mode, seed=seed, metric="score")
