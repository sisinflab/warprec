from typing import TypeVar, Dict, Type, Optional, Callable, List, Generic, TYPE_CHECKING

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

T = TypeVar("T")


class BasicRegistry(Generic[T]):
    """Basic registry with functionality to store information.

    Args:
        registry_name (str): Name of the registry.
    """

    def __init__(self, registry_name: str):
        self._registry: Dict[str, Type[T]] = {}
        self.registry_name = registry_name

    def register(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to register a class in the registry.

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
            """
            nonlocal name
            key = (name or cls.__name__).upper()
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, name: str, *args, **kwargs) -> T:
        """
        Get an instance from the registry by name.
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

    def list_registered(self) -> List[str]:
        """List all registered names.

        Returns:
            List[str]: The list of names stored.
        """
        return list(self._registry.keys())


# Singleton basic registries
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


class ModelRegistry:
    """The definition of a registry for Recommendation Models.

    Args:
        registry_name (str): The name of the registry.
    """

    def __init__(self, registry_name: str):
        self._registry: Dict[str, Dict[str, "Recommender"]] = {}
        self.registry_name = registry_name

    def register(
        self, name: Optional[str] = None, implementation: Optional[str] = None
    ) -> Callable:
        """
        Decorator to register a class in the registry.

        Args:
            name (Optional[str]): Name for registration. If None, uses class name.
            implementation (Optional[str]): The implementation of the model.
                If None, uses latest.

        Returns:
            Callable: The decorator to register new data.
        """

        def decorator(cls: "Recommender") -> "Recommender":
            """The definition of the decorator.

            Args:
                cls (Recommender): A recommender class.

            Returns:
                Recommender: A recommender class.
            """
            key = (name or cls.name).upper()
            imp = (implementation or "latest").lower()

            # Ensure key exists in registry
            if key not in self._registry:
                self._registry[key] = {}

            self._registry[key][imp] = cls
            return cls

        return decorator

    def get(
        self, name: str, *args, implementation: str = None, **kwargs
    ) -> "Recommender":
        """
        Get an instance from the registry by name.

        Args:
            name (str): Name of the registered recommender.
            *args: Arguments to pass to the class constructor.
            implementation (str): The name of the implementation of the recommender.
            **kwargs: Keyword arguments to pass to the class constructor.

        Returns:
            Recommender: The recommender model.

        Raises:
            ValueError: If name is not to be found in registry.
        """
        if implementation is None:
            implementation = "latest"
        versions = self._registry.get(name.upper())
        if versions is None:
            raise ValueError(
                f"'{name}' not found in {self.registry_name} registry. "
                f"Available options: {list(self._registry.keys())}"
            )
        cls = versions.get(implementation.lower())
        if cls is None:
            raise ValueError(
                f"'{implementation}' not found in {name} registered implementations. "
                f"Available options: {list(versions.keys())}"
            )
        return cls(*args, **kwargs)

    def list_registered(self) -> List[str]:
        """List all registered names.

        Returns:
            List[str]: The list of names registered.
        """
        return list(self._registry.keys())

    def list_implementations(self, model_name: str) -> List[str]:
        """List all implementations of given model.

        Args:
            model_name (str): Name of the model.

        Returns:
            List[str]: The list of implementations available
                for the given model.
        """
        return list(self._registry[model_name].keys())


# Singleton class for model registry
model_registry = ModelRegistry("Models")
