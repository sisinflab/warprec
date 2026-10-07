"""Behavioural tests for user code brought in through general.custom_modules.

A custom module is how a user adds a model, a metric or a filter without
touching WarpRec, so it has to load from wherever the configuration points,
and before anything in the configuration asks for the names it registers.
"""

import uuid
from pathlib import Path

import pandas as pd
import pytest
import yaml

from warprec.utils.config.config import load_design_configuration
from warprec.utils.helpers import load_custom_modules
from warprec.utils.registry import metric_registry

METRIC_SOURCE = """
from warprec.evaluation.metrics.accuracy.precision import Precision
from warprec.utils.registry import metric_registry


@metric_registry.register("{name}")
class {name}(Precision):
    pass
"""


def write_metric_module(directory: Path) -> tuple[Path, str]:
    """Write a module registering a metric under a name no other test uses.

    Args:
        directory (Path): Where to write the module.

    Returns:
        tuple[Path, str]: The module's path and the metric name it registers.
    """
    name = f"Metric{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name.lower()}.py"
    path.write_text(METRIC_SOURCE.format(name=name))
    return path, name


def test_a_module_given_by_absolute_path_is_loaded(tmp_path: Path):
    """The documented form is a path, so a path must be enough to load it."""
    path, name = write_metric_module(tmp_path / "user_code")

    load_custom_modules([str(path)])

    assert name.upper() in metric_registry.list_registered()


def test_a_module_given_by_relative_path_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A path relative to where WarpRec runs, as the guides write it."""
    monkeypatch.chdir(tmp_path)
    _, name = write_metric_module(tmp_path / "guides" / "custom")

    load_custom_modules([f"guides/custom/{name.lower()}.py"])

    assert name.upper() in metric_registry.list_registered()


def test_a_package_given_by_path_is_loaded(tmp_path: Path):
    """A directory holding an __init__.py is a module too."""
    package = tmp_path / "user_code" / f"pkg_{uuid.uuid4().hex[:8]}"
    _, name = write_metric_module(package)
    (package / "__init__.py").write_text(f"from .{name.lower()} import {name}\n")

    load_custom_modules([str(package)])

    assert name.upper() in metric_registry.list_registered()


def test_a_module_that_fails_to_import_stops_the_run(tmp_path: Path):
    """Carrying on without the user's code only fails later, and less clearly."""
    broken = tmp_path / f"broken_{uuid.uuid4().hex[:8]}.py"
    broken.write_text("import a_package_that_does_not_exist\n")

    with pytest.raises(ImportError, match=broken.stem):
        load_custom_modules([str(broken)])


def test_a_custom_metric_can_be_configured_with_parameters(tmp_path: Path):
    """complex_metrics checks its names, so the module must already be loaded."""
    path, name = write_metric_module(tmp_path / "user_code")
    data = tmp_path / "data.tsv"
    pd.DataFrame(
        {"user_id": [1, 1, 2, 2], "item_id": [1, 2, 1, 3], "timestamp": [1, 2, 3, 4]}
    ).to_csv(data, sep="\t", index=False)
    config = tmp_path / "config.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "reader": {
                    "loading_strategy": "dataset",
                    "data_type": "transaction",
                    "reading_method": "local",
                    "local_path": str(data),
                    "rating_type": "implicit",
                },
                "splitter": {
                    "test_splitting": {"strategy": "temporal_holdout", "ratio": 0.5}
                },
                "models": {"Pop": {}},
                "evaluation": {
                    "top_k": [2],
                    "metrics": ["nDCG"],
                    "complex_metrics": [{"name": name, "params": {}}],
                },
                "general": {"custom_modules": [str(path)]},
            }
        )
    )

    loaded = load_design_configuration(str(config))

    assert loaded.evaluation.complex_metrics[0].name == name
