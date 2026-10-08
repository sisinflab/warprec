"""The optimization block reaches the trainer of every train-pipeline trial.

A trial rebuilds its model configuration from the hyperparameters Ray Tune
sampled, which carry no optimization block. Precision and gradient clipping
therefore have to travel with the rest of the trial's data, or the trial
trains with neither while the design pipeline and the retraining apply both.
"""

from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict

import pytest

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.trainer import objectives
from warprec.recommenders.trainer import trainer as trainer_module
from warprec.utils.config.evaluation_configuration import EvaluationConfig
from warprec.utils.registry import params_registry

HYPERPARAMETERS = {
    "embedding_size": 8,
    "reg_weight": 1e-5,
    "batch_size": 64,
    "epochs": 1,
    "learning_rate": 0.001,
}
OPTIMIZATION = {
    "precision": "64-true",
    "gradient_clip": 0.5,
    "gradient_clip_algorithm": "value",
    "num_workers": 0,
    "max_concurrent_trials": 1,
}


class RecordingTrainer:
    """Stands in for the Lightning trainer and keeps what it was built with."""

    built: Dict[str, Any] = {}

    def __init__(self, **kwargs: Any):
        RecordingTrainer.built = kwargs

    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Train nothing."""


def _bundle(monkeypatch: pytest.MonkeyPatch, dataset: Dataset, tmp_path: Path) -> dict:
    """The data bundle the train pipeline hands to every trial of BPR.

    Args:
        monkeypatch (pytest.MonkeyPatch): Keeps Ray out of the test.
        dataset (Dataset): The dataset to optimise on.
        tmp_path (Path): Where Ray Tune would store its results.

    Returns:
        dict: The bundle, merged with the trial's sampled hyperparameters the
            way the trial driver merges them.
    """
    captured: Dict[str, Any] = {}
    monkeypatch.setattr(trainer_module.ray, "put", lambda value: value)
    monkeypatch.setattr(
        trainer_module.tune,
        "with_parameters",
        lambda fn, **kwargs: captured.update(kwargs) or fn,
    )
    monkeypatch.setattr(trainer_module.tune, "with_resources", lambda fn, resources: fn)
    monkeypatch.setattr(
        trainer_module.Trainer, "_build_or_restore_tuner", lambda self, **kw: None
    )

    params = params_registry.get("BPR", **HYPERPARAMETERS, optimization=OPTIMIZATION)
    trainer_module.Trainer(storage_path=str(tmp_path))._setup_tuner(
        model_name="BPR",
        params=params,
        dataset=dataset,
        evaluation=EvaluationConfig(
            top_k=[5], metrics=["nDCG"], validation_metric="nDCG@5"
        ),
        validation_score="nDCG@5",
        device="cpu",
        ray_verbose=0,
    )
    return {**captured["data_bundle"], "params": dict(HYPERPARAMETERS)}


def test_a_trial_trains_with_the_configured_precision_and_clipping(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset, tmp_path
):
    """The trial's trainer gets the block the configuration asked for."""
    config = _bundle(monkeypatch, dataset, tmp_path)

    # Run the trial in this process, without Ray Train around it
    monkeypatch.setattr(objectives.ray, "get", lambda value: value)
    monkeypatch.setattr(
        objectives.train,
        "get_context",
        lambda: SimpleNamespace(get_world_size=lambda: 1),
    )
    monkeypatch.setattr(objectives.train, "get_checkpoint", lambda: None)
    monkeypatch.setattr(objectives.train, "report", lambda *a, **k: None)
    monkeypatch.setattr(objectives.L, "Trainer", RecordingTrainer)
    monkeypatch.setattr(objectives, "RayTrainReportCallback", lambda: None)
    RecordingTrainer.built = {}

    objectives.objective_function(config)

    assert RecordingTrainer.built, "the trial never built a trainer"
    assert RecordingTrainer.built["precision"] == "64-true"
    assert RecordingTrainer.built["gradient_clip_val"] == pytest.approx(0.5)
    assert RecordingTrainer.built["gradient_clip_algorithm"] == "value"
