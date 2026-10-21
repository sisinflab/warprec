"""
This script contains the wrapper of the Ray wrappers for the search algorithms.
At the time of writing this there is no common interface provided by Ray.
This makes the process of registering these classes not very 'pythonic',
but it serves its purpose. In future this class must be refactored if possible.

TODO: Refactor this script in a more pythonic way.

Author: Avolio Marco
Date: 03/03/2025
"""

from typing import Any, Optional
from abc import ABC, abstractmethod

from optuna.samplers import TPESampler
from ray.tune import Tuner
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bohb import TuneBOHB
from warprec.utils.enums import SearchAlgorithms
from warprec.utils.logger import logger
from warprec.utils.registry import search_algorithm_registry


class CappedVariantGenerator(BasicVariantGenerator):
    """Ray Tune's grid and random searcher, keeping the cap it was last given.

    Ray restores a paused search's searcher with the cap it was paused with.
    WarpRec computes the cap from the cluster as the search resumes, and sets
    it on the restored searcher, which then keeps it.
    """

    def set_state(self, state):
        cap = self.max_concurrent
        super().set_state(state)
        self.max_concurrent = cap


class ConcurrencyCap(ConcurrencyLimiter):
    """Ray Tune's concurrency limiter, keeping the cap it was last given.

    See 'CappedVariantGenerator'.
    """

    def set_state(self, state):
        cap = self.max_concurrent
        super().set_state(state)
        self.max_concurrent = cap
        self._set_searcher_max_concurrency()


def recap_restored_search(tuner: Tuner, max_concurrent_trials: int) -> None:
    """Give a restored search the concurrency cap computed for this session.

    Ray Tune restores the searcher a search was paused with, cap included, and
    'Tuner.restore' takes no new configuration, so the cap is set on the restored
    searcher. A search paused by WarpRec 2.2.5 or earlier has Ray's own searcher,
    which is converted to the one that keeps the cap.

    Args:
        tuner (Tuner): The restored tuner, not fitted yet.
        max_concurrent_trials (int): The cap.
    """
    # Tuner.restore exposes no configuration, so the restored one is reached
    tune_config = getattr(getattr(tuner, "_local_tuner", None), "_tune_config", None)
    if tune_config is None:
        logger.attention(
            "The restored search keeps the concurrency it was paused with: "
            "this Ray version does not expose its configuration."
        )
        return

    searcher = tune_config.search_alg
    if isinstance(searcher, BasicVariantGenerator):
        if type(searcher) is BasicVariantGenerator:
            searcher.__class__ = CappedVariantGenerator
        searcher.max_concurrent = max_concurrent_trials
    elif isinstance(searcher, ConcurrencyCap):
        searcher.set_max_concurrency(max_concurrent_trials)
        searcher._set_searcher_max_concurrency()
    elif searcher is not None:
        tune_config.search_alg = ConcurrencyCap(
            searcher, max_concurrent=max_concurrent_trials
        )
    tune_config.max_concurrent_trials = None
    logger.msg(
        f"The restored search runs up to {max_concurrent_trials} trials at once."
    )


class BaseSearchWrapper(ABC):
    """Common interface for all search algorithm wrappers."""

    @abstractmethod
    def __init__(self, **kwargs: Any):
        pass


@search_algorithm_registry.register(SearchAlgorithms.GRID)
class GridSearchWrapper(BaseSearchWrapper):
    """Wrapper for the GridSearch algorithm in Ray Tune.

    Returns the searcher Ray Tune uses by default, seeded. A grid space is
    enumerated deterministically, but the search space is chosen per parameter:
    a 'choice' entry under this strategy is sampled, and would be unseeded
    without the explicit searcher.

    Args:
        seed (Optional[int]): The seed to make the experiment reproducible.
        max_concurrent_trials (Optional[int]): The concurrency cap. Ray only
            applies it to a searcher it builds itself, so it is set here.
        **kwargs (Any): Keyword arguments.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_concurrent_trials: Optional[int] = None,
        **kwargs: Any,
    ):
        pass

    def __new__(
        cls,
        seed: Optional[int] = None,
        max_concurrent_trials: Optional[int] = None,
        **kwargs: Any,
    ):
        return CappedVariantGenerator(
            random_state=seed, max_concurrent=max_concurrent_trials or 0
        )


@search_algorithm_registry.register(SearchAlgorithms.RANDOM)
class RandomSearchWrapper(BaseSearchWrapper):
    """Wrapper for the RandomSearch algorithm in Ray Tune.

    Returns the searcher Ray Tune uses by default, seeded. Leaving Ray to
    build it makes the sampling unreproducible, since its random state is
    then never set.

    Args:
        seed (Optional[int]): The seed to make the experiment reproducible.
        max_concurrent_trials (Optional[int]): The concurrency cap. Ray only
            applies it to a searcher it builds itself, so it is set here.
        **kwargs (Any): Keyword arguments.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_concurrent_trials: Optional[int] = None,
        **kwargs: Any,
    ):
        pass

    def __new__(
        cls,
        seed: Optional[int] = None,
        max_concurrent_trials: Optional[int] = None,
        **kwargs: Any,
    ):
        return CappedVariantGenerator(
            random_state=seed, max_concurrent=max_concurrent_trials or 0
        )


@search_algorithm_registry.register(SearchAlgorithms.HYPEROPT)
class HyperOptWrapper(HyperOptSearch, BaseSearchWrapper):
    """Wrapper for the HyperOpt algorithm in Ray Tune.

    Args:
        mode (str): The mode to run the optimization. Must be
            either 'min' or 'max'.
        metric (str): The metric to optimize.
        seed (int): The seed to make the experiment reproducible.
        n_startup_trials (Optional[int]): The number of random trials to run
            before the Parzen estimators start guiding the search. When None,
            the HyperOpt default is kept.
        **kwargs (Any): Keyword arguments.
    """

    def __init__(
        self,
        mode: str,
        metric: str,
        seed: int,
        n_startup_trials: Optional[int] = None,
        **kwargs: Any,
    ):
        if n_startup_trials is None:
            super().__init__(mode=mode, metric=metric, random_state_seed=seed)
            return

        super().__init__(
            mode=mode,
            metric=metric,
            random_state_seed=seed,
            n_initial_points=n_startup_trials,
        )


@search_algorithm_registry.register(SearchAlgorithms.OPTUNA)
class OptunaWrapper(OptunaSearch, BaseSearchWrapper):
    """Wrapper for the Optuna algorithm in Ray Tune.

    Args:
        mode (str): The mode to run the optimization. Must be
            either 'min' or 'max'.
        metric (str): The metric to optimize.
        seed (int): The seed to make the experiment reproducible.
        n_startup_trials (Optional[int]): The number of random trials to run
            before the Parzen estimators start guiding the search. When None,
            the Optuna default is kept.
        **kwargs (Any): Keyword arguments.
    """

    def __init__(
        self,
        mode: str,
        metric: str,
        seed: int,
        n_startup_trials: Optional[int] = None,
        **kwargs: Any,
    ):
        if n_startup_trials is None:
            super().__init__(mode=mode, metric=metric, seed=seed)
            return

        super().__init__(
            mode=mode,
            metric=metric,
            sampler=TPESampler(seed=seed, n_startup_trials=n_startup_trials),
        )


@search_algorithm_registry.register(SearchAlgorithms.BOHB)
class BOHBWrapper(TuneBOHB, BaseSearchWrapper):
    """Wrapper for the BOHB algorithm in Ray Tune.

    This is the Bayesian half of BOHB and is meant to be paired with the 'bohb'
    scheduler, which provides the HyperBand half.

    Args:
        mode (str): The mode to run the optimization. Must be
            either 'min' or 'max'.
        metric (str): The metric to optimize.
        seed (int): The seed to make the experiment reproducible.
        **kwargs (Any): Keyword arguments.
    """

    def __init__(self, mode: str, metric: str, seed: int, **kwargs: Any):
        super().__init__(mode=mode, metric=metric, seed=seed)
