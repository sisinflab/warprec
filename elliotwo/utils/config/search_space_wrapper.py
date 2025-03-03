# pylint: disable=too-few-public-methods
from typing import Callable

from ray import tune
from elliotwo.utils.enums import SearchSpace
from elliotwo.utils.registry import search_space_registry


class SearchSpaceWrapper:
    """The Wrapper interface for Search Spaces supported by Ray Tune."""

    def __call__(self, *args):
        raise NotImplementedError


@search_space_registry.register(SearchSpace.GRID)
class GridSpaceWrapper(SearchSpaceWrapper):
    """The Grid Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.choice(*args)


@search_space_registry.register(SearchSpace.CHOICE)
class ChoiceSpaceWrapper(SearchSpaceWrapper):
    """The Choice Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.choice(*args)


@search_space_registry.register(SearchSpace.UNIFORM)
class UniformSpaceWrapper(SearchSpaceWrapper):
    """The Uniform Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.uniform(*args)


@search_space_registry.register(SearchSpace.QUNIFORM)
class QUniformSpaceWrapper(SearchSpaceWrapper):
    """The QUniform Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.quniform(*args)


@search_space_registry.register(SearchSpace.LOGUNIFORM)
class LogUniformSpaceWrapper(SearchSpaceWrapper):
    """The LogUniform Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.loguniform(*args)


@search_space_registry.register(SearchSpace.QLOGUNIFORM)
class QLogUniformSpaceWrapper(SearchSpaceWrapper):
    """The QLogUniform Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.qloguniform(*args)


@search_space_registry.register(SearchSpace.RANDN)
class RandnSpaceWrapper(SearchSpaceWrapper):
    """The Randn Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.randn(*args)


@search_space_registry.register(SearchSpace.QRANDN)
class QRandnSpaceWrapper(SearchSpaceWrapper):
    """The QRandn Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.qrandn(*args)


@search_space_registry.register(SearchSpace.RANDINT)
class RandintSpaceWrapper(SearchSpaceWrapper):
    """The Randint Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.randint(*args)


@search_space_registry.register(SearchSpace.QRANDINT)
class QRandintSpaceWrapper(SearchSpaceWrapper):
    """The QRandint Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.qrandint(*args)


@search_space_registry.register(SearchSpace.LOGRANDINT)
class LogRandintSpaceWrapper(SearchSpaceWrapper):
    """The LogRandint Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.lograndint(*args)


@search_space_registry.register(SearchSpace.QLOGRANDINT)
class QLogRandintSpaceWrapper(SearchSpaceWrapper):
    """The QLogRandint Search Space Wrapper."""

    def __call__(self, *args) -> Callable:
        return tune.qlograndint(*args)
