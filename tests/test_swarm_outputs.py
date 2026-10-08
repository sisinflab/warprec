"""Behavioural tests for what the swarm pipeline leaves behind.

The swarm runs every model at once, but it is still a run of the same
experiment, so it must write the same files the train pipeline writes and
compare every model it evaluated. The models themselves run on the Ray cluster;
here they are stood in for, so that what is under test is the driver: what it
does with each finished model.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import ray
import torch
import yaml

from conftest import make_model
from warprec.data.dataset import Dataset
from warprec.pipelines import common, swarm

MODELS = ["EASE", "ItemKNN"]


def write_configuration(tmp_path: Path) -> Path:
    """Write a swarm configuration whose outputs land under the temporary path.

    Args:
        tmp_path (Path): The temporary directory pytest provides.

    Returns:
        Path: The configuration file.
    """
    data = tmp_path / "data.tsv"
    data.write_text("user_id\titem_id\trating\ttimestamp\n1\t2\t3\t4\n")
    config = {
        "reader": {
            "loading_strategy": "dataset",
            "data_type": "transaction",
            "reading_method": "local",
            "local_path": str(data),
            "rating_type": "implicit",
            "sep": "\t",
        },
        "writer": {
            "dataset_name": "ds",
            "writing_method": "local",
            "local_experiment_path": str(tmp_path / "experiment"),
        },
        "splitter": {"test_splitting": {"strategy": "temporal_holdout", "ratio": 0.1}},
        "models": {
            "EASE": {"l2": 10, "meta": {"save_model": True}},
            "ItemKNN": {"k": 10, "similarity": "cosine", "meta": {"save_model": True}},
        },
        "evaluation": {
            "top_k": [10],
            "metrics": ["nDCG"],
            "save_per_user": True,
            "stat_significance": {"wilcoxon_test": True},
        },
        "general": {"time_report": True},
        "run": {"name": "swarm-run"},
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return path


def finished_model(model_name: str, dataset: Dataset) -> tuple:
    """What a model's remote pipeline hands back to the driver once finished.

    Args:
        model_name (str): The name of the model.
        dataset (Dataset): The dataset the model is built on.

    Returns:
        tuple: The same tuple 'remote_model_pipeline' returns.
    """
    model = make_model(model_name, dataset)
    generator = torch.Generator().manual_seed(len(model_name))
    results = {10: {"nDCG": torch.rand(dataset.info()["n_users"], generator=generator)}}
    ray_report = {
        "Trainable Params (Best Model)": 0,
        "Total Params (Best Model)": 0,
    }
    timing = {
        "Model Name": model_name,
        "Data Preparation Time": 1.0,
        "Hyperparameter Exploration Time": 2.0,
        **ray_report,
        "Evaluation Time": 3.0,
        "Inference Time": 0.001,
        "Total Time": 6.0,
    }
    params = {
        model_name: {"Best Params": model.get_params(), "Best Training Iteration": 1}
    }
    return model_name, model, results, params, timing, ray_report


@pytest.fixture
def swarm_run(
    tmp_path: Path, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Run the swarm driver with the cluster stood in for.

    Args:
        tmp_path (Path): The temporary directory pytest provides.
        dataset (Dataset): The shared synthetic dataset.
        monkeypatch (pytest.MonkeyPatch): Replaces Ray and the remote tasks.

    Returns:
        Path: The experiment directory the run wrote to.
    """
    monkeypatch.setattr(common.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(swarm, "prepare_datasets", lambda context: (dataset, None, []))

    def wait(futures: List[Any], num_returns: int = 1, timeout: float = 0.0):
        return futures[:num_returns], futures[num_returns:]

    monkeypatch.setattr(
        swarm,
        "ray",
        SimpleNamespace(
            put=lambda value: value,
            get=lambda value: value,
            wait=wait,
            cancel=lambda *args, **kwargs: None,
            exceptions=ray.exceptions,
        ),
    )
    monkeypatch.setattr(
        swarm,
        "remote_model_pipeline",
        SimpleNamespace(
            remote=lambda model_name, **kwargs: finished_model(model_name, dataset)
        ),
    )

    swarm.swarm_pipeline(str(write_configuration(tmp_path)))
    return tmp_path / "experiment" / "ds"


def only(directory: Path, pattern: str) -> Path:
    """The single file in a directory matching a pattern.

    Args:
        directory (Path): The directory to look in.
        pattern (str): The glob pattern.

    Returns:
        Path: The matching file.
    """
    matches = list(directory.glob(pattern))
    assert len(matches) == 1, (
        f"{pattern}: {sorted(p.name for p in directory.iterdir())}"
    )
    return matches[0]


def test_the_swarm_writes_the_results_of_every_model(swarm_run: Path):
    """Overall and per-user results exist for every model, as in a train run."""
    evaluation = swarm_run / "evaluation"

    overall = only(evaluation, "Overall_Results_*.tsv").read_text()
    for model_name in MODELS:
        assert model_name in overall
        only(evaluation, f"{model_name}_k_10_per_user_*.tsv")


def test_the_swarm_writes_the_best_parameters(swarm_run: Path):
    """The parameter file lists every model with its best training iteration."""
    params = only(swarm_run / "params", "Overall_Params_*.json")

    content: Dict[str, Any] = yaml.safe_load(params.read_text())
    assert set(content) == set(MODELS)
    assert all("Best Training Iteration" in entry for entry in content.values())


def test_the_swarm_saves_the_models_it_was_asked_to(swarm_run: Path):
    """meta.save_model is honoured for every model."""
    saved = sorted(p.name for p in (swarm_run / "serialized").glob("*.pth"))

    assert len(saved) == len(MODELS)
    for model_name in MODELS:
        assert any(name.startswith(model_name) for name in saved)


def test_the_swarm_writes_the_time_report(swarm_run: Path):
    """The time report covers every model."""
    report = only(swarm_run / "evaluation", "Time_Report_*.tsv").read_text()

    for model_name in MODELS:
        assert model_name in report


def test_the_swarm_compares_every_model_it_evaluated(swarm_run: Path):
    """Significance runs over the models of this run, not only earlier ones."""
    table = only(swarm_run / "evaluation", "Wilcoxon_test_*.tsv").read_text()

    for model_name in MODELS:
        assert model_name in table
