import difflib
from typing import (
    Any,
    TypeVar,
    Dict,
    Type,
    Optional,
    Callable,
    List,
    Generic,
    TYPE_CHECKING,
)

from pydantic import ValidationError
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer

from warprec.errors import ConfigurationError, RegistryError

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
        config_section (Optional[str]): The configuration section the registered
            classes validate, e.g. 'models' for the parameter registry. When
            given, a pydantic ValidationError raised building one is raised as
            a ConfigurationError whose keys sit under '<section>.<name>'.
    """

    def __init__(self, registry_name: str, config_section: Optional[str] = None):
        self._registry: Dict[str, Type[T]] = {}
        self.registry_name = registry_name
        self.config_section = config_section

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
                RegistryError: If a different class is registered under the name.
            """
            nonlocal name
            key = (name or cls.__name__).upper()
            existing = self._registry.get(key)
            # The same class registered again, because its module was imported
            # again (a notebook cell run twice, a Ray worker), replaces itself.
            if existing is not None and _qualified(existing) != _qualified(cls):
                raise RegistryError(
                    f"Cannot register {_qualified(cls)} as '{name or cls.__name__}' "
                    f"in the {self.registry_name} registry: the name is already "
                    f"registered to {_qualified(existing)} (names are compared "
                    "ignoring case). Register the class under another name.",
                    registry=self.registry_name,
                    name=name or cls.__name__,
                    available=self.list_registered(),
                )
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, name: str, *args, **kwargs) -> T:
        """Get an instance from the registry by name.

        An unknown name raises the RegistryError of get_class.

        Args:
            name (str): Name of the registered class.
            *args: Arguments to pass to the class constructor.
            **kwargs: Keyword arguments to pass to the class constructor.

        Returns:
            T: Any type of object stored previously.

        Raises:
            ConfigurationError: If the registry validates a configuration
                section and the arguments are not valid for the class.
        """
        cls = self.get_class(name)
        if self.config_section is None:
            return cls(*args, **kwargs)
        try:
            return cls(*args, **kwargs)
        except ValidationError as error:
            message, errors = ConfigurationError.describe_validation(
                error, "valid", prefix=f"{self.config_section}.{name}"
            )
            raise ConfigurationError(message, errors=errors) from error

    def get_class(self, name: str) -> Type[T]:
        """Get the class from the registry by name.

        Args:
            name (str): Name of the registered class.

        Returns:
            Type[T]: Any type of object stored previously.

        Raises:
            RegistryError: If name is not to be found in registry.
        """
        cls = self._registry.get(name.upper())
        if cls is None:
            raise RegistryError(
                **self.unknown(
                    name,
                    f"'{name}' not found in {self.registry_name} registry. "
                    f"Available options: {list(self._registry.keys())}",
                )
            )
        return cls

    def unknown(self, name: str, message: str) -> Dict[str, Any]:
        """The arguments of the RegistryError for a name that is not registered.

        Raise it as ``RegistryError(**registry.unknown(name, message))``.

        Args:
            name (str): The unknown name.
            message (str): What is wrong; the closest registered names are
                appended to it when any is close.

        Returns:
            Dict[str, Any]: The message, the registry, the name and the names
                registered.
        """
        hint = self.suggest(name)
        if hint:
            message += f" {hint}" if message.endswith((".", "?")) else f". {hint}"
        return {
            "message": message,
            "registry": self.registry_name,
            "name": name,
            "available": self.list_registered(),
        }

    def suggest(self, name: str) -> str:
        """The hint naming the registered names closest to an unknown one.

        Args:
            name (str): The unknown name.

        Returns:
            str: "Did you mean 'X'?", naming the class as it is spelled when
                its name is the registered one, or an empty string when no
                registered name is close.
        """
        close = difflib.get_close_matches(name.upper(), list(self._registry), n=3)
        if not close:
            return ""
        spelled = []
        for key in close:
            cls_name = self._registry[key].__name__
            spelled.append(cls_name if cls_name.upper() == key else key)
        return f"Did you mean {', '.join(repr(s) for s in spelled)}?"

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
params_registry: BasicRegistry["RecomModel"] = BasicRegistry(
    "Params", config_section="models"
)
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
