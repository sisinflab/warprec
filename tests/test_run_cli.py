"""python -m warprec.run reports an error the user can fix without a traceback.

A WarpRecError is printed as '<ClassName>: <message>' on stderr and the process
exits with status 1; --debug raises it so the traceback shows.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from warprec.errors import ConfigurationError
from warprec.run import main


def write_config(tmp_path: Path, frame: pd.DataFrame, **changes: Any) -> Path:
    """A design configuration over the shared interactions, written to disk.

    Args:
        tmp_path (Path): Where to write the data and the configuration.
        frame (pd.DataFrame): The interactions to read.
        **changes (Any): Top-level blocks replacing the defaults.

    Returns:
        Path: The configuration's path.
    """
    data = tmp_path / "data.tsv"
    frame[["user_id", "item_id", "rating", "timestamp"]].to_csv(
        data, sep="\t", index=False
    )
    config = {
        "reader": {
            "loading_strategy": "dataset",
            "data_type": "transaction",
            "reading_method": "local",
            "local_path": str(data),
            "rating_type": "implicit",
            "sep": "\t",
        },
        "splitter": {"test_splitting": {"strategy": "temporal_holdout", "ratio": 0.2}},
        "models": {"Pop": {}},
        "evaluation": {"top_k": [5], "metrics": ["nDCG"]},
        "general": {"device": "cpu"},
        **changes,
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_a_missing_configuration_file_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture
):
    with pytest.raises(SystemExit) as stop:
        main(["-c", str(tmp_path / "missing.yml"), "-p", "design"])
    assert stop.value.code == 1
    assert capsys.readouterr().err == (
        "DataNotFoundError: Configuration file not found at: "
        f"{tmp_path / 'missing.yml'}\n"
    )


def test_debug_raises_the_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="missing.yml"):
        main(["-c", str(tmp_path / "missing.yml"), "-p", "design", "--debug"])


def test_an_unknown_pipeline_is_a_configuration_error(
    tmp_path: Path, interactions_frame: pd.DataFrame, capsys: pytest.CaptureFixture
):
    path = write_config(tmp_path, interactions_frame)
    with pytest.raises(SystemExit):
        main(["-c", str(path), "-p", "tune"])
    assert capsys.readouterr().err.startswith(
        "ConfigurationError: Invalid pipeline specified: tune."
    )
    with pytest.raises(ValueError, match="Invalid pipeline"):
        main(["-c", str(path), "-p", "tune", "--debug"])


def test_an_invalid_configuration_lists_each_key(
    tmp_path: Path, interactions_frame: pd.DataFrame, capsys: pytest.CaptureFixture
):
    path = write_config(
        tmp_path,
        interactions_frame,
        splitter={"test_splitting": {"strategy": "temporal_holdout"}},
        evaluation={"top_k": [5], "metrics": ["nDCG"], "seed": -1},
    )
    with pytest.raises(SystemExit):
        main(["-c", str(path), "-p", "design"])
    err = capsys.readouterr().err
    assert err.startswith(
        f"ConfigurationError: {path} is not a valid design configuration:\n"
    )
    assert (
        "\n  splitter.test_splitting: You have chosen temporal_holdout splitting "
        "but the 'ratio' field has not been filled.\n"
    ) in err
    assert "\n  evaluation.seed: " in err
    assert "Value error" not in err
    assert "input_value" not in err and "errors.pydantic.dev" not in err


def test_the_error_keeps_the_keys_and_the_pydantic_cause(
    tmp_path: Path, interactions_frame: pd.DataFrame
):
    path = write_config(
        tmp_path,
        interactions_frame,
        splitter={"test_splitting": {"strategy": "temporal_holdout"}},
    )
    with pytest.raises(ConfigurationError) as caught:
        main(["-c", str(path), "-p", "design", "--debug"])
    assert caught.value.path == str(path)
    assert [key for key, _ in caught.value.errors] == ["splitter.test_splitting"]
    assert type(caught.value.__cause__).__name__ == "ValidationError"


def test_the_module_prints_no_traceback(tmp_path: Path):
    done = subprocess.run(
        [sys.executable, "-m", "warprec.run", "-c", str(tmp_path / "x.yml")]
        + ["-p", "design"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert done.stderr.strip().endswith(
        f"DataNotFoundError: Configuration file not found at: {tmp_path / 'x.yml'}"
    )
