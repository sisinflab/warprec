"""
This script contains the wrapper of the Ray wrappers for the search algorithms.
At the time of writing this there is no common interface provided by Ray.
This makes the process of registering these classes not very 'pythonic',
but it serves its purpose. In future this class must be refactored if possible.

TODO: Refactor this script in a more pythonic way.

Author: Avolio Marco
Date: 03/03/2025
"""

import copy
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from optuna.samplers import TPESampler
from ray.tune import Tuner
from ray.tune.search import ConcurrencyLimiter, Searcher
from ray.tune.search.util import _set_search_properties_backwards_compatible
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bohb import TuneBOHB
from warprec.utils.enums import SearchAlgorithms
from warprec.utils.helpers import model_param_from_dict
from warprec.utils.logger import logger
from warprec.utils.registry import search_algorithm_registry


def invalid_combination(
    model_name: Optional[str], config: Dict[str, Any]
) -> Optional[str]:
    """Why a sampled combination of hyperparameters cannot be trained.

    The check a trial makes before it builds its model, made before the trial
    is launched: a model whose hyperparameters constrain each other (an
    embedding size the number of attention heads must divide) refuses some of
    the combinations its search space holds.

    Args:
        model_name (Optional[str]): The model searched, None to check nothing.
        config (Dict[str, Any]): The sampled hyperparameters.

    Returns:
        Optional[str]: Why the combination is invalid, None when it is valid.
    """
    if model_name is None:
        return None
    try:
        params = model_param_from_dict(model_name, dict(config))
        if params.need_single_trial_validation:
            params.validate_single_trial_params()
    except ValueError as error:
        return str(error)
    return None


class CappedVariantGenerator(BasicVariantGenerator):
    """Ray Tune's grid and random searcher, keeping the cap it was last given.

    Ray restores a paused search's searcher with the cap it was paused with.
    WarpRec computes the cap from the cluster as the search resumes, and sets
    it on the restored searcher, which then keeps it.

    Given a model name, it also skips the combinations of hyperparameters the
    model refuses, instead of launching a trial that could only fail.
    """

    model_name: Optional[str] = None

    def set_state(self, state):
        cap = self.max_concurrent
        super().set_state(state)
        self.max_concurrent = cap

    def next_trial(self):
        while True:
            trial = super().next_trial()
            if trial is None:
                return None
            reason = invalid_combination(self.model_name, trial.config)
            if reason is None:
                return trial
            self._live_trials.discard(trial.trial_id)
            logger.attention(
                f"Skipped the combination {trial.config} of {self.model_name}, "
                f"which cannot be trained: {reason}"
            )


class ValidSuggestions(Searcher):
    """A searcher that suggests only combinations the model can be trained with.

    A combination the model refuses is reported to the searcher as a failed
    trial, so that a model-based searcher learns to avoid it, and another is
    asked for in its place, under an id of its own: a searcher answers the same
    id with the same combination.

    Args:
        searcher (Searcher): The searcher to wrap.
        model_name (str): The model searched.

    Attributes:
        MAX_ATTEMPTS (int): How many combinations to try before letting one
            the model refuses through, to fail as a trial.
    """

    MAX_ATTEMPTS: int = 100

    def __init__(self, searcher: Searcher, model_name: str):
        self.searcher = searcher
        self.model_name = model_name
        # The id the searcher knows each trial by, when it is not the trial's
        self.inner_ids: Dict[str, str] = {}
        super().__init__(metric=searcher.metric, mode=searcher.mode)

    def _inner(self, trial_id: str) -> str:
        return self.inner_ids.get(trial_id, trial_id)

    def set_search_properties(
        self, metric: Optional[str], mode: Optional[str], config: Dict, **spec: Any
    ) -> bool:
        return _set_search_properties_backwards_compatible(
            self.searcher.set_search_properties, metric, mode, config, **spec
        )

    def set_max_concurrency(self, max_concurrent: int) -> bool:
        return self.searcher.set_max_concurrency(max_concurrent)

    def suggest(self, trial_id: str) -> Optional[Dict]:
        inner_id = trial_id
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            suggestion = self.searcher.suggest(inner_id)
            if suggestion in (None, Searcher.FINISHED):
                return suggestion
            reason = invalid_combination(self.model_name, suggestion)
            if reason is None or attempt == self.MAX_ATTEMPTS:
                break
            logger.attention(
                f"Skipped the combination {suggestion} of {self.model_name}, "
                f"which cannot be trained: {reason}"
            )
            self.searcher.on_trial_complete(inner_id, result=None, error=True)
            inner_id = f"{trial_id}_retry{attempt}"
        if inner_id != trial_id:
            self.inner_ids[trial_id] = inner_id
        return suggestion

    def on_trial_result(self, trial_id: str, result: Dict) -> None:
        self.searcher.on_trial_result(self._inner(trial_id), result)

    def on_trial_complete(
        self, trial_id: str, result: Optional[Dict] = None, error: bool = False
    ) -> None:
        self.searcher.on_trial_complete(
            self.inner_ids.pop(trial_id, trial_id), result=result, error=error
        )

    def add_evaluated_point(
        self,
        parameters: Dict,
        value: float,
        error: bool = False,
        pruned: bool = False,
        intermediate_values: Optional[List[float]] = None,
    ) -> None:
        self.searcher.add_evaluated_point(
            parameters, value, error, pruned, intermediate_values
        )

    def get_state(self) -> Dict:
        state = self.__dict__.copy()
        del state["searcher"]
        return copy.deepcopy(state)

    def set_state(self, state: Dict) -> None:
        self.__dict__.update(state)

    def save(self, checkpoint_path: str) -> None:
        self.searcher.save(checkpoint_path)

    def restore(self, checkpoint_path: str) -> None:
        self.searcher.restore(checkpoint_path)

    # BOHB pauses and resumes trials through its searcher
    def on_pause(self, trial_id: str) -> None:
        self.searcher.on_pause(self._inner(trial_id))  # type: ignore[attr-defined]

    def on_unpause(self, trial_id: str) -> None:
        self.searcher.on_unpause(self._inner(trial_id))  # type: ignore[attr-defined]


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
