from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from pydantic import ValidationError


class WarpRecError(Exception):
    """The base of every error WarpRec raises for something the user can fix.

    Catching it catches every error below, and nothing else: a failure inside
    WarpRec itself stays a RuntimeError or an AssertionError, and RunPaused,
    which pauses a run rather than failing it, is not one of them.

    Each class below also derives from the built-in exception WarpRec raised
    before it existed (ValueError for most of them), so an ``except ValueError``
    written against an earlier release still catches what it caught then.

    Every error pickles to an equal one, fields included, so it crosses a Ray
    task boundary as itself: a trial that raises a ConfigurationError is caught
    on the driver by ``except ConfigurationError``.
    """


class ConfigurationError(WarpRecError, ValueError):
    """The configuration is invalid or contradicts itself.

    Raised for a configuration file, and for the objects built from one: model
    hyperparameters, search spaces, optimizer and scheduler names, filter and
    splitter parameters. Loading a configuration file raises a single
    ConfigurationError that lists every invalid key.

    Args:
        message (str): What is wrong and how to fix it.
        path (Optional[Union[str, Path]]): The configuration file, when the
            error comes from loading one.
        errors (Optional[List[Tuple[str, str]]]): One (dotted key path, message)
            pair per invalid key, when the error comes from validating them.

    Both are kept as attributes of the same name: ``path`` as a string or None,
    ``errors`` as a list, empty when not given.
    """

    def __init__(
        self,
        message: str,
        path: Optional[Union[str, Path]] = None,
        errors: Optional[List[Tuple[str, str]]] = None,
    ):
        super().__init__(message)
        self.path = str(path) if path is not None else None
        self.errors = list(errors) if errors is not None else []

    @classmethod
    def from_validation_error(
        cls,
        error: "ValidationError",
        what: str,
        path: Optional[Union[str, Path]] = None,
        prefix: str = "",
    ) -> "ConfigurationError":
        """Turn a pydantic ValidationError into one readable ConfigurationError.

        Each invalid key is listed on its own line as its dotted path and the
        validator's own message, without pydantic's input dump and links. An
        error raised by a nested validation that already produced a
        ConfigurationError (a model's hyperparameters, validated inside the
        configuration) contributes its own keys rather than the outer one.

        Args:
            error (ValidationError): The pydantic error.
            what (str): What failed to validate, e.g. 'a valid train configuration'.
            path (Optional[Union[str, Path]]): The configuration file, if any.
            prefix (str): The key path the validated object sits under, e.g.
                'models.BPR', prepended to each key.

        Returns:
            ConfigurationError: The error, to be raised from the ValidationError.
        """
        errors = _validation_errors(error, prefix)
        subject = str(path) if path is not None else (prefix or error.title)
        lines = "\n".join(
            f"  {key}: {msg}" if key else f"  {msg}" for key, msg in errors
        )
        return cls(f"{subject} is not {what}:\n{lines}", path=path, errors=errors)


def _validation_errors(error: "ValidationError", prefix: str) -> List[Tuple[str, str]]:
    """The (dotted key path, message) pairs of a pydantic ValidationError.

    Args:
        error (ValidationError): The pydantic error.
        prefix (str): The key path prepended to each key.

    Returns:
        List[Tuple[str, str]]: One pair per invalid key.
    """
    pairs: List[Tuple[str, str]] = []
    for detail in error.errors(include_url=False, include_input=False):
        cause = (detail.get("ctx") or {}).get("error")
        if isinstance(cause, ConfigurationError) and cause.errors:
            pairs.extend(cause.errors)
            continue
        key = ".".join(
            str(part) for part in (prefix, *detail["loc"]) if part not in ("", None)
        )
        message = detail["msg"]
        for marker in ("Value error, ", "Assertion failed, "):
            message = message.removeprefix(marker)
        pairs.append((key, message))
    return pairs


class DataError(WarpRecError, ValueError):
    """The data cannot support what was asked of it.

    For example a missing column, a dataset left empty by filtering, a user
    with too few unseen items to draw sampled negatives from, or side
    information that does not match the dataset.
    """


class DataNotFoundError(DataError, FileNotFoundError):
    """A file WarpRec was asked to read does not exist.

    A dataset, a side-information file, a split, a configuration file or a
    checkpoint file. It is also a FileNotFoundError.
    """


class RegistryError(WarpRecError, ValueError):
    """A name is not registered, or is already registered to another class.

    When a registered name is close to the unknown one, the message ends with
    "Did you mean 'X'?".

    Args:
        message (str): What is wrong and how to fix it.
        registry (Optional[str]): The name of the registry.
        name (Optional[str]): The name looked up or registered.
        available (Optional[List[str]]): The names registered.

    All three are kept as attributes of the same name, ``available`` as a list.
    """

    def __init__(
        self,
        message: str,
        registry: Optional[str] = None,
        name: Optional[str] = None,
        available: Optional[List[str]] = None,
    ):
        super().__init__(message)
        self.registry = registry
        self.name = name
        self.available = list(available) if available is not None else []


class ExtensionError(WarpRecError, ValueError):
    """User code breaks a contract WarpRec relies on.

    For example a stash key already taken, a custom metric returning a result
    of the wrong type or reading a configuration block that is missing, or a
    hook given arguments that cannot be pickled.
    """


class CustomModuleError(ExtensionError, ImportError):
    """A custom module cannot be imported, or its name is taken by another one.

    It is also an ImportError.
    """


class ModelError(WarpRecError, ValueError):
    """A model cannot be built or used as asked.

    For example a model built for another number of users or items, a model
    that needs side information or context it was not given, or a space
    estimate asked without the interactions it is computed from.
    """


class CheckpointError(ModelError):
    """A checkpoint cannot be used: it holds another model or another dataset.

    A checkpoint file that does not exist raises DataNotFoundError, which is
    also a FileNotFoundError, as WarpRec did before this class existed.

    Args:
        message (str): What is wrong and how to fix it.
        path (Optional[Union[str, Path]]): The checkpoint.

    The path is kept as the ``path`` attribute, a string or None.
    """

    def __init__(self, message: str, path: Optional[Union[str, Path]] = None):
        super().__init__(message)
        self.path = str(path) if path is not None else None


class EvaluationError(WarpRecError, ValueError):
    """A metric or an evaluation strategy cannot run on what it is given.

    For example an unsupported strategy, a metric that needs the training
    data, propensities or clusters it was not given, or a multi-objective
    metric asked to combine a global metric.
    """


class RunStateError(WarpRecError, ValueError):
    """Resuming a run conflicts with the state it saved."""


class ClusterError(WarpRecError, ConnectionError):
    """The Ray cluster cannot be reached. It is also a ConnectionError."""


class ServingError(WarpRecError):
    """A request the model cannot answer, with the HTTP status that says why.

    It crosses the Ray boundary as a plain dictionary rather than as a raised
    exception, so that the gateway maps it to a status code without depending
    on how Ray wraps exceptions raised inside a replica.

    Args:
        status (int): The HTTP status of the failure.
        detail (str): The message for the caller.

    Both are kept as attributes of the same name.
    """

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail

    def __reduce__(self) -> Tuple[Any, ...]:
        """Pickle the error with both of its constructor arguments.

        Returns:
            Tuple[Any, ...]: The class, its arguments and its state.
        """
        return (type(self), (self.status, self.detail), self.__dict__)

    def to_dict(self) -> Dict[str, Any]:
        """The error as the gateway expects it.

        Returns:
            Dict[str, Any]: The status and the detail.
        """
        return {"status": self.status, "detail": self.detail}


__all__ = [
    "WarpRecError",
    "ConfigurationError",
    "DataError",
    "DataNotFoundError",
    "RegistryError",
    "ExtensionError",
    "CustomModuleError",
    "ModelError",
    "CheckpointError",
    "EvaluationError",
    "RunStateError",
    "ClusterError",
    "ServingError",
]
