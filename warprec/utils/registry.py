from typing import TypeVar, Dict, Type, Optional, Callable, List, Generic, TYPE_CHECKING

from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer

if TYPE_CHECKING:
    from warprec.recommenders.base_recommender import Recommender
    from warprec.utils.config import RecomModel, SearchSpaceWrapper
    from warprec.evaluation.metrics.base_metric import BaseMetric
    from warprec.data.splitting.strategies import SplittingStrategy
    from warprec.recommenders.trainer.search_algorithm_wrapper import (
        BaseSearchWrapper,
    )
    from warprec.recommenders.trainer.scheduler_wrapper import (
        BaseSchedulerWrapper,
    )
    from warprec.recommenders.similarities import Similarity
    from warprec.data.filtering import Filter
    from warprec.recommenders.reranking.base import Reranker
    from warprec.evaluation.statistical_significance import StatisticalTest

T = TypeVar("T")


def _qualified(cls: type) -> str:
    """The module and qualified name of a class, which identify it across imports.

    Args:
        cls (type): The class.

    Returns:
        str: The dotted name, e.g. 'warprec.recommenders.EASE'.
    """
    return f"{cls.__module__}.{cls.__qualname__}"


class BasicRegistry(Generic[T]):
    """Basic registry with functionality to store information.

    Args:
        registry_name (str): Name of the registry.
    """

    def __init__(self, registry_name: str):
        self._registry: Dict[str, Type[T]] = {}
        self.registry_name = registry_name

    def register(self, name: Optional[str] = None) -> Callable:
        """Decorator to register a class in the registry.

        Args:
            name (Optional[str]): Name for registration. If None, uses class name.

        Returns:
            Callable: The decorator to register new data.
        """

        def decorator(cls: Type[T]) -> Type[T]:
            """The definition of the decorator.

            Args:
                cls (Type[T]): Any type of class to be stored.

            Returns:
                Type[T]: Any type of class.

            Raises:
                ValueError: If a different class is registered under the name.
            """
            nonlocal name
            key = (name or cls.__name__).upper()
            existing = self._registry.get(key)
            # The same class registered again, because its module was imported
            # again (a notebook cell run twice, a Ray worker), replaces itself.
            if existing is not None and _qualified(existing) != _qualified(cls):
                raise ValueError(
                    f"Cannot register {_qualified(cls)} as '{name or cls.__name__}' "
                    f"in the {self.registry_name} registry: the name is already "
                    f"registered to {_qualified(existing)} (names are compared "
                    "ignoring case). Register the class under another name."
                )
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, name: str, *args, **kwargs) -> T:
        """Get an instance from the registry by name.

        Args:
            name (str): Name of the registered class.
            *args: Arguments to pass to the class constructor.
            **kwargs: Keyword arguments to pass to the class constructor.

        Returns:
            T: Any type of object stored previously.

        Raises:
            ValueError: If name is not to be found in registry.
        """
        cls = self._registry.get(name.upper())
        if cls is None:
            raise ValueError(
                f"'{name}' not found in {self.registry_name} registry. "
                f"Available options: {list(self._registry.keys())}"
            )
        return cls(*args, **kwargs)

    def get_class(self, name: str) -> Type[T]:
        """Get the class from the registry by name.

        Args:
            name (str): Name of the registered class.

        Returns:
            Type[T]: Any type of object stored previously.

        Raises:
            ValueError: If name is not to be found in registry.
        """
        cls = self._registry.get(name.upper())
        if cls is None:
            raise ValueError(
                f"'{name}' not found in {self.registry_name} registry. "
                f"Available options: {list(self._registry.keys())}"
            )
        return cls

    def registered_name(self, cls: type) -> Optional[str]:
        """Get the name a class was registered under.

        A class is registered under the name given to the decorator, which need
        not be its own, so it is looked up by identity. A copy of the class made
        by importing its module again is recognised by its module and qualified
        name.

        Args:
            cls (type): The class to look up.

        Returns:
            Optional[str]: The upper-case registry key, or None when the class
                is not registered.
        """
        for key, registered in self._registry.items():
            if registered is cls:
                return key
        for key, registered in self._registry.items():
            if _qualified(registered) == _qualified(cls):
                return key
        return None

    def list_registered(self) -> List[str]:
        """List all registered names.

        Returns:
            List[str]: The list of names stored.
        """
        return list(self._registry.keys())


# Singleton basic registries
model_registry: BasicRegistry["Recommender"] = BasicRegistry("Recommender")
metric_registry: BasicRegistry["BaseMetric"] = BasicRegistry("Metrics")
splitting_registry: BasicRegistry["SplittingStrategy"] = BasicRegistry("Splitting")
params_registry: BasicRegistry["RecomModel"] = BasicRegistry("Params")
search_algorithm_registry: BasicRegistry["BaseSearchWrapper"] = BasicRegistry(
    "SearchAlgorithms"
)
scheduler_registry: BasicRegistry["BaseSchedulerWrapper"] = BasicRegistry("Schedulers")
search_space_registry: BasicRegistry["SearchSpaceWrapper"] = BasicRegistry(
    "SearchSpace"
)
similarities_registry: BasicRegistry["Similarity"] = BasicRegistry("Similarity")
filter_registry: BasicRegistry["Filter"] = BasicRegistry("Filter")
reranker_registry: BasicRegistry["Reranker"] = BasicRegistry("Reranker")
stat_significance_registry: BasicRegistry["StatisticalTest"] = BasicRegistry(
    "StatisticalSignificance"
)
lr_scheduler_registry: BasicRegistry[LRScheduler] = BasicRegistry("LRScheduler")
optimizer_registry: BasicRegistry[Optimizer] = BasicRegistry("Optimizer")
