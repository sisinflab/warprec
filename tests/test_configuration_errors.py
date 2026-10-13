"""A configuration that cannot run raises a ConfigurationError naming each key.

The errors raised inside the validators keep their own class (RegistryError
for an unknown name, DataNotFoundError for a missing file), reachable through
the pydantic error the ConfigurationError is raised from.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from warprec.errors import (
    ConfigurationError,
    DataNotFoundError,
    RegistryError,
)
import warprec.recommenders  # noqa: F401  (populates the optimizer registry)
from warprec.utils.config.model_configuration import OptimizerConfig
from warprec.utils.config.config import (
    DesignConfiguration,
    load_design_configuration,
    load_train_configuration,
)

from test_run_cli import write_config

BPR = {
    "embedding_size": 8,
    "reg_weight": 0.0,
    "batch_size": 8,
    "epochs": 1,
    "learning_rate": 0.01,
}


def causes(error: ConfigurationError) -> list:
    """The errors the validators raised, behind a ConfigurationError.

    Args:
        error (ConfigurationError): The error a loader raised.

    Returns:
        list: The exception each invalid key was reported with.
    """
    assert isinstance(error.__cause__, ValidationError)
    return [detail["ctx"]["error"] for detail in error.__cause__.errors()]


def test_an_unknown_model_suggests_the_registered_one(
    tmp_path: Path, interactions_frame: pd.DataFrame
):
    path = write_config(tmp_path, interactions_frame, models={"BRP": BPR})
    with pytest.raises(ConfigurationError) as caught:
        load_design_configuration(str(path))
    assert str(caught.value).endswith("Did you mean 'BPR'?")
    (cause,) = causes(caught.value)
    assert isinstance(cause, RegistryError)
    assert (cause.registry, cause.name) == ("Recommender", "BRP")


def test_an_invalid_hyperparameter_is_reported_under_its_model(
    tmp_path: Path, interactions_frame: pd.DataFrame
):
    path = write_config(
        tmp_path, interactions_frame, models={"BPR": {**BPR, "learning_rate": -1}}
    )
    with pytest.raises(ConfigurationError) as caught:
        load_design_configuration(str(path))
    assert [key for key, _ in caught.value.errors] == ["models.BPR.learning_rate"]
    assert "\n  models.BPR.learning_rate: Values of learning_rate" in str(caught.value)


def test_an_unknown_optimizer_is_a_registry_error():
    with pytest.raises(ValidationError) as caught:
        OptimizerConfig(name="Adma")
    cause = caught.value.errors()[0]["ctx"]["error"]
    assert isinstance(cause, RegistryError)
    assert (cause.registry, cause.name) == ("Optimizer", "Adma")
    assert ". Did you mean 'ADAM'" in str(cause)


def test_missing_training_data_is_a_data_not_found_error(
    tmp_path: Path, interactions_frame: pd.DataFrame
):
    path = write_config(tmp_path, interactions_frame)
    (tmp_path / "data.tsv").unlink()
    with pytest.raises(DataNotFoundError, match="Training file not at"):
        load_design_configuration(str(path))
    # Built directly, the configuration still raises the built-in
    config = yaml.safe_load(path.read_text())
    with pytest.raises(FileNotFoundError) as caught:
        DesignConfiguration(**config)
    assert not isinstance(caught.value, DataNotFoundError)


def test_a_missing_configuration_file_is_a_data_not_found_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as caught:
        load_train_configuration(str(tmp_path / "missing.yml"))
    assert isinstance(caught.value, DataNotFoundError)
