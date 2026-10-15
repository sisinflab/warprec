"""What the time report and a trial's results say about where the time went.

A trial's wall clock mixes starting the Ray Train workers, building the data
loaders and the model, training, and the evaluation that ends every epoch. The
report has to tell those apart: how long a model takes to train, to complete
an epoch, to be validated, and above all the time spent only training.
"""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import lightning as L
import pytest

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.callbacks import WarpRecLightningIntegrationCallback
from warprec.recommenders.trainer import objectives
from warprec.recommenders.trainer import trainer as trainer_module
from warprec.recommenders.trainer.timing import (
    EPOCH_SAMPLES,
    EPOCH_TRAIN_S,
    EPOCH_VALIDATION_S,
    SETUP_S,
    TRAIN_SAMPLES_PER_S,
    TRIAL_STARTED_AT,
)
from warprec.utils.config.evaluation_configuration import EvaluationConfig
from warprec.utils.registry import params_registry

BPR = {
    "embedding_size": 8,
    "reg_weight": 1e-5,
    "batch_size": 64,
    "epochs": 3,
    "learning_rate": 0.001,
}
EPOCH_KEYS = (EPOCH_TRAIN_S, EPOCH_VALIDATION_S, EPOCH_SAMPLES, TRAIN_SAMPLES_PER_S)
SLOW_VALIDATION_S = 0.3
STARTED_BEFORE_S = 1.0


class RecordingReport(L.Callback):
    """Stands in for Ray's report callback: keeps what each epoch would report."""

    reports: List[Dict[str, float]] = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Record the metrics Ray Train would receive for this epoch.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The model.
        """
        RecordingReport.reports.append(
            {key: value.item() for key, value in trainer.callback_metrics.items()}
        )


def _trial_config(
    monkeypatch: pytest.MonkeyPatch,
    dataset: Dataset,
    tmp_path: Path,
    model_name: str,
    hyperparameters: Dict[str, Any],
    eval_every_n: int = 1,
) -> dict:
    """The configuration one trial of the train pipeline receives.

    Args:
        monkeypatch (pytest.MonkeyPatch): Keeps Ray out of the test.
        dataset (Dataset): The dataset to optimise on.
        tmp_path (Path): Where Ray Tune would store its results.
        model_name (str): The model of the trial.
        hyperparameters (Dict[str, Any]): The trial's hyperparameters.
        eval_every_n (int): How often the trial validates.

    Returns:
        dict: The configuration, as the trial driver hands it to the worker,
            with a trial that started a second ago.
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

    params = params_registry.get(
        model_name,
        **hyperparameters,
        optimization={
            "num_workers": 0,
            "max_concurrent_trials": 1,
            "eval_every_n": eval_every_n,
        },
    )
    trainer_module.Trainer(storage_path=str(tmp_path))._setup_tuner(
        model_name=model_name,
        params=params,
        dataset=dataset,
        evaluation=EvaluationConfig(
            top_k=[5], metrics=["nDCG"], validation_metric="nDCG@5"
        ),
        validation_score="nDCG@5",
        device="cpu",
        ray_verbose=0,
    )

    # Run the trial in this process, without Ray Train around it
    monkeypatch.setattr(objectives.ray, "get", lambda value: value)
    monkeypatch.setattr(
        objectives.train,
        "get_context",
        lambda: SimpleNamespace(get_world_size=lambda: 1),
    )
    monkeypatch.setattr(objectives.train, "get_checkpoint", lambda: None)
    return {
        **captured["data_bundle"],
        "params": dict(hyperparameters),
        TRIAL_STARTED_AT: time.time() - STARTED_BEFORE_S,
    }


def test_every_epoch_reports_its_training_and_its_validation_apart(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset, tmp_path: Path
):
    """Training, validation, samples and setup reach every epoch's report."""
    config = _trial_config(monkeypatch, dataset, tmp_path, "BPR", BPR, eval_every_n=2)
    monkeypatch.setattr(objectives, "RayTrainReportCallback", RecordingReport)
    RecordingReport.reports = []

    evaluate = WarpRecLightningIntegrationCallback.on_validation_epoch_end

    def slow_validation(self: Any, trainer: L.Trainer, pl_module: Any) -> None:
        """Validate, after waiting long enough to be told from training.

        Args:
            self (Any): The integration callback.
            trainer (L.Trainer): The trainer.
            pl_module (Any): The model.
        """
        time.sleep(SLOW_VALIDATION_S)
        evaluate(self, trainer, pl_module)

    monkeypatch.setattr(
        WarpRecLightningIntegrationCallback, "on_validation_epoch_end", slow_validation
    )

    objectives.objective_function(config)

    reports = RecordingReport.reports
    assert len(reports) == BPR["epochs"]
    training_examples = len(dataset.train_set)
    for epoch, report in enumerate(reports):
        assert all(key in report for key in (*EPOCH_KEYS, SETUP_S)), report
        validated = (epoch + 1) % 2 == 0
        if validated:
            assert report[EPOCH_VALIDATION_S] >= SLOW_VALIDATION_S
        else:
            assert report[EPOCH_VALIDATION_S] == 0.0
        # The wait inside the validation is not training time
        assert 0.0 < report[EPOCH_TRAIN_S] < SLOW_VALIDATION_S
        assert report[EPOCH_SAMPLES] == training_examples
        assert report[TRAIN_SAMPLES_PER_S] == pytest.approx(
            report[EPOCH_SAMPLES] / report[EPOCH_TRAIN_S]
        )
        assert report[SETUP_S] >= STARTED_BEFORE_S
        assert report[SETUP_S] == reports[0][SETUP_S]


def test_a_closed_form_trial_reports_its_fit_as_its_one_epoch(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset, tmp_path: Path
):
    """EASE's fit is its training, its evaluation its validation."""
    config = _trial_config(monkeypatch, dataset, tmp_path, "EASE", {"l2": 10.0})
    reports: List[Dict[str, float]] = []
    monkeypatch.setattr(
        objectives.train, "report", lambda metrics, **kwargs: reports.append(metrics)
    )

    objectives.objective_function(config)

    assert len(reports) == 1
    report = reports[0]
    assert report[EPOCH_TRAIN_S] > 0.0
    assert report[EPOCH_VALIDATION_S] > 0.0
    assert report[EPOCH_SAMPLES] == len(dataset.train_set)
    assert report[TRAIN_SAMPLES_PER_S] == pytest.approx(
        report[EPOCH_SAMPLES] / report[EPOCH_TRAIN_S]
    )
    assert report[SETUP_S] >= STARTED_BEFORE_S
