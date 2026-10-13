"""The errors WarpRec raises for something the user can fix.

Each one is a WarpRecError, and also the built-in exception WarpRec raised
before it existed, so handlers written against earlier releases keep working.
Each one pickles to an equal error, so it survives a Ray task boundary.
"""

import pickle

import pytest
import ray

import warprec
from warprec import errors
from warprec.errors import (
    CheckpointError,
    ClusterError,
    ConfigurationError,
    CustomModuleError,
    DataError,
    DataNotFoundError,
    EvaluationError,
    ExtensionError,
    ModelError,
    RegistryError,
    RunStateError,
    ServingError,
    WarpRecError,
)
from warprec.utils.pause import RunPaused

# Every class, an instance with its fields, and the built-ins it must still be
EXAMPLES = [
    (
        ConfigurationError(
            "c.yml is not valid", path="c.yml", errors=[("a.b", "is wrong")]
        ),
        (ValueError,),
    ),
    (DataError("no column"), (ValueError,)),
    (DataNotFoundError("no file"), (ValueError, FileNotFoundError, DataError)),
    (
        RegistryError("unknown", registry="Recommender", name="BRP", available=["BPR"]),
        (ValueError,),
    ),
    (ExtensionError("contract"), (ValueError,)),
    (CustomModuleError("import"), (ValueError, ImportError, ExtensionError)),
    (ModelError("model"), (ValueError,)),
    (CheckpointError("other model", path="m.pth"), (ValueError, ModelError)),
    (EvaluationError("metric"), (ValueError,)),
    (RunStateError("resume"), (ValueError,)),
    (ClusterError("cluster"), (ConnectionError,)),
    (ServingError(404, "unknown item"), ()),
]


def fields(error: Exception) -> tuple:
    """What makes two errors equal: class, message, arguments and fields.

    Args:
        error (Exception): The error.

    Returns:
        tuple: The comparable parts of the error.
    """
    return type(error), str(error), error.args, vars(error)


@pytest.mark.parametrize(
    ("error", "builtins"), EXAMPLES, ids=lambda e: type(e).__name__
)
def test_every_error_is_a_warprec_error_and_its_builtin(
    error: WarpRecError, builtins: tuple
):
    assert isinstance(error, WarpRecError)
    for builtin in builtins:
        assert isinstance(error, builtin)


@pytest.mark.parametrize(
    ("error", "builtins"), EXAMPLES, ids=lambda e: type(e).__name__
)
def test_every_error_pickles_to_an_equal_one(error: WarpRecError, builtins: tuple):
    assert fields(pickle.loads(pickle.dumps(error))) == fields(error)


def test_the_structured_fields_are_kept():
    error = RegistryError(
        "unknown", registry="Recommender", name="BRP", available=["BPR"]
    )
    assert (error.registry, error.name, error.available) == (
        "Recommender",
        "BRP",
        ["BPR"],
    )
    assert CheckpointError("other", path="m.pth").path == "m.pth"
    config = ConfigurationError("bad", path="c.yml", errors=[("a", "b")])
    assert (config.path, config.errors) == ("c.yml", [("a", "b")])
    serving = ServingError(404, "unknown item")
    assert (serving.status, serving.detail) == (404, "unknown item")
    assert serving.to_dict() == {"status": 404, "detail": "unknown item"}


def test_every_class_is_exported():
    assert set(errors.__all__) == {
        name for name in dir(errors) if name.endswith("Error") and name[0] != "_"
    }
    assert "errors" in warprec.__all__
    assert warprec.errors is errors


def test_the_serving_error_keeps_its_import_path():
    from warprec.serving.errors import ServingError as Moved

    assert Moved is ServingError


def test_a_paused_run_is_not_an_error():
    assert not issubclass(RunPaused, WarpRecError)


@ray.remote
def _raise_configuration_error() -> None:
    """Raise a ConfigurationError inside a Ray task.

    Raises:
        ConfigurationError: Always.
    """
    raise ConfigurationError("c.yml is not valid", path="c.yml", errors=[("a", "b")])


def test_an_error_crosses_a_ray_task_boundary_as_itself():
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=1, include_dashboard=False)
    try:
        with pytest.raises(ConfigurationError, match="c.yml is not valid") as caught:
            ray.get(_raise_configuration_error.remote())
    finally:
        if started:
            ray.shutdown()
    assert isinstance(caught.value, ValueError)
    assert caught.value.path == "c.yml"
    assert caught.value.errors == [("a", "b")]
