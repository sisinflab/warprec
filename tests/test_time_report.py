"""What the time report and a trial's results say about where the time went.

A trial's wall clock mixes starting the Ray Train workers, building the data
loaders and the model, training, and the evaluation that ends every epoch. The
report has to tell those apart: how long a model takes to train, to complete
an epoch, to be validated, and above all the time spent only training.
"""

import math
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.data.timing import DataPreparationTimes
from warprec.data.writer import LocalWriter
from warprec.pipelines.common import model_timing_row
from warprec.pipelines.remotes import ml
from warprec.pipelines.remotes.ml import remote_model_retraining
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
    TrialSpans,
    TrialTiming,
    search_overhead,
    summarise_timing,
    trial_timing,
)
from warprec.utils.config.evaluation_configuration import EvaluationConfig
from warprec.utils.registry import params_registry

from conftest import make_model

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


class Paused(BaseException):
    """Stands for the trial process being stopped by a pause.

    A BaseException, so that the trial's own error handling lets it through,
    as it lets through the end of the process.
    """


def _session(
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
    folder: Path,
    started_before_s: float,
    pause_after: Optional[int] = None,
    resume_from: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Run one session of a trial and return the reports Ray Tune receives.

    Like Ray's report callback, every epoch saves its checkpoint and then
    reports. A pause stops the process between the two, so the paused epoch
    is saved but never reported.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces Ray Train's session.
        config (dict): The trial configuration.
        folder (Path): Where the session saves its checkpoints.
        started_before_s (float): How long before now the session started.
        pause_after (Optional[int]): The epoch the session pauses after.
        resume_from (Optional[int]): The epoch whose checkpoint it resumes from.

    Returns:
        List[Dict[str, float]]: The reports, in the order Ray Tune gets them.
    """
    reports: List[Dict[str, float]] = []

    class Report(L.Callback):
        """Saves the epoch's checkpoint, then reports its metrics."""

        CHECKPOINT_NAME = "checkpoint.ckpt"

        def on_train_epoch_end(
            self, trainer: L.Trainer, pl_module: L.LightningModule
        ) -> None:
            """Save, pause when due, report.

            Args:
                trainer (L.Trainer): The trainer.
                pl_module (L.LightningModule): The model.

            Raises:
                Paused: When the session pauses after this epoch.
            """
            target = folder / str(trainer.current_epoch)
            target.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(target / self.CHECKPOINT_NAME, weights_only=False)
            if trainer.current_epoch == pause_after:
                raise Paused()
            report = {k: v.item() for k, v in trainer.callback_metrics.items()}
            reports.append({**report, "epoch": trainer.current_epoch})

    checkpoint = (
        SimpleNamespace(
            as_directory=lambda: nullcontext(str(folder / str(resume_from)))
        )
        if resume_from is not None
        else None
    )
    monkeypatch.setattr(objectives.train, "get_checkpoint", lambda: checkpoint)
    monkeypatch.setattr(
        objectives.train, "report", lambda metrics, **kwargs: reports.append(metrics)
    )
    monkeypatch.setattr(objectives, "RayTrainReportCallback", Report)
    session = {
        **config,
        TRIAL_STARTED_AT: time.time() - started_before_s,
        "session_id": f"started {started_before_s}s ago",
        "last_logged_epoch": None if resume_from is None else resume_from - 1,
    }
    try:
        objectives.objective_function(session)
    except Paused:
        pass
    return reports


def test_a_resumed_trial_reports_each_epoch_with_the_setup_of_its_session(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset, tmp_path: Path
):
    """An epoch carries the setup of the session that trained it.

    The epoch a pause fell on is reported by the session that resumes, from
    its checkpoint: it was trained, timed and set up by the paused session.
    """
    hyperparameters = {**BPR, "epochs": 4}
    config = _trial_config(monkeypatch, dataset, tmp_path, "BPR", hyperparameters)
    first_setup, second_setup = 100.0, 200.0

    first = _session(monkeypatch, config, tmp_path, first_setup, pause_after=1)
    # The paused epoch's report, as the paused session logged it
    paused_epoch = torch.load(tmp_path / "1" / "checkpoint.ckpt", weights_only=False)[
        "callbacks"
    ]["WarpRecLightningIntegrationCallback"]["last_report"]
    second = _session(monkeypatch, config, tmp_path, second_setup, resume_from=1)

    assert [report["epoch"] for report in first] == [0]
    assert [report["epoch"] for report in second] == [1, 2, 3]
    assert first[0][SETUP_S] == pytest.approx(first_setup, abs=30.0)
    assert second[0] == paused_epoch
    assert second[0][SETUP_S] == pytest.approx(first_setup, abs=30.0)
    for report in second[1:]:
        assert report[SETUP_S] == pytest.approx(second_setup, abs=30.0)


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


def _history(
    train: List[float],
    validation: List[float],
    setup: float,
    samples: float = 100.0,
    epochs: Optional[List[int]] = None,
) -> pd.DataFrame:
    """The reports of one trial, as Ray Tune keeps them.

    Args:
        train (List[float]): The training time of each epoch.
        validation (List[float]): The validation time of each epoch.
        setup (float): The trial's setup.
        samples (float): The training examples of every epoch.
        epochs (Optional[List[int]]): The epoch of each report, when it is
            not simply its position.

    Returns:
        pd.DataFrame: The trial's history.
    """
    return pd.DataFrame(
        {
            "epoch": epochs if epochs is not None else list(range(len(train))),
            EPOCH_TRAIN_S: train,
            EPOCH_VALIDATION_S: validation,
            EPOCH_SAMPLES: [samples] * len(train),
            SETUP_S: [setup] * len(train),
        }
    )


def test_a_trial_history_gives_its_timing_once_per_epoch():
    """An epoch reported twice (a resumed trial) is counted once."""
    timing = trial_timing(
        _history([1.0, 2.0, 2.5, 3.0], [0.5, 0.0, 0.5, 0.5], 4.0, epochs=[0, 1, 1, 2])
    )

    assert timing is not None
    assert timing.epoch_train_s == [1.0, 2.5, 3.0]
    assert timing.epoch_validation_s == [0.5, 0.5, 0.5]
    assert timing.setup_s == 4.0
    assert timing.samples == 300.0


def test_a_history_without_timing_gives_none():
    """A trial that failed before training reports no timing at all."""
    assert trial_timing(pd.DataFrame({"nDCG@10": [-math.inf]})) is None


def test_the_time_report_describes_the_best_trial_and_every_trial():
    """Every new column of the time report, from the trials' histories."""
    trials = {
        "best": trial_timing(_history([2.0, 4.0, 3.0], [1.0, 0.0, 1.0], 10.0)),
        "other": trial_timing(_history([1.0], [2.0], 6.0)),
        "failed": None,
    }

    report = summarise_timing(trials, ["best"])

    assert report == pytest.approx(
        {
            "Best Trial Setup Time (s)": 10.0,
            "Best Trial Train-Only Time (s)": 9.0,
            "Best Trial Validation Time (s)": 2.0,
            "Best Trial Training Time (s)": 11.0,
            "Best Trial Epochs": 3,
            "Best Trial Mean Epoch Train Time (s)": 3.0,
            "Best Trial Min Epoch Train Time (s)": 2.0,
            "Best Trial Max Epoch Train Time (s)": 4.0,
            "Best Trial Mean Epoch Validation Time (s)": 1.0,
            "Best Trial Train Throughput (samples/s)": 300.0 / 9.0,
            "Mean Trial Setup Time (s)": 8.0,
            "Mean Trial Train-Only Time (s)": 5.0,
            "Mean Trial Validation Time (s)": 2.0,
        }
    )


def test_the_best_trials_of_a_cross_validation_are_averaged_over_the_folds():
    """Under cross-validation the best configuration ran once per fold."""
    trials = {
        "fold0": trial_timing(_history([2.0, 2.0], [1.0, 1.0], 10.0)),
        "fold1": trial_timing(_history([4.0], [3.0], 20.0)),
    }

    report = summarise_timing(trials, ["fold0", "fold1"])

    assert report["Best Trial Setup Time (s)"] == pytest.approx(15.0)
    assert report["Best Trial Train-Only Time (s)"] == pytest.approx(4.0)
    assert report["Best Trial Epochs"] == pytest.approx(1.5)
    assert report["Best Trial Max Epoch Train Time (s)"] == pytest.approx(4.0)
    assert report["Best Trial Train Throughput (samples/s)"] == pytest.approx(
        300.0 / 8.0
    )


def test_a_report_without_timing_leaves_the_columns_empty():
    """No trial reported timing: the columns are there, and empty."""
    report = summarise_timing({"failed": None}, ["failed"])

    assert "Best Trial Train-Only Time (s)" in report
    assert all(np.isnan(value) for value in report.values())


def _spans(*intervals: tuple) -> TrialSpans:
    """Trial spans as the driver would have recorded them.

    Args:
        *intervals (tuple): (trial id, start, end) of each run of a trial.

    Returns:
        TrialSpans: The spans.
    """
    spans = TrialSpans()
    for trial_id, start, end in intervals:
        spans.spans.setdefault(trial_id, []).append((start, end))
    return spans


def test_the_search_overhead_is_what_trials_did_not_spend_working():
    """Trials one at a time: the wall clock minus their setup and training."""
    trials = {
        "a": TrialTiming(setup_s=3.0, epoch_train_s=[2.0], epoch_validation_s=[1.0]),
        "b": TrialTiming(setup_s=2.0, epoch_train_s=[1.0], epoch_validation_s=[1.0]),
    }
    spans = _spans(("a", 0.0, 7.0), ("b", 7.5, 12.0))

    assert search_overhead(20.0, spans, trials) == pytest.approx(10.0)


@pytest.mark.parametrize(
    "intervals, trials",
    [
        # Two trials at once: their times overlap and cannot be subtracted
        ([("a", 0.0, 7.0), ("b", 5.0, 12.0)], ["a", "b"]),
        # A trial without timing: its time is neither overhead nor work
        ([("a", 0.0, 7.0), ("b", 7.5, 12.0)], ["a", "b", "failed"]),
        # A trial the driver did not see start (a restored search)
        ([("a", 0.0, 7.0)], ["a", "b"]),
        # A trial paused and run again counts its setup once
        ([("a", 0.0, 3.0), ("b", 3.5, 6.0), ("a", 6.5, 9.0)], ["a", "b"]),
    ],
)
def test_the_search_overhead_is_not_given_when_it_cannot_be_honest(
    intervals: List[tuple], trials: List[str]
):
    """Overlapping, untimed or unseen trials leave the overhead empty."""
    timing = TrialTiming(setup_s=1.0, epoch_train_s=[1.0], epoch_validation_s=[1.0])
    by_trial: Dict[str, Optional[TrialTiming]] = {
        trial_id: None if trial_id == "failed" else timing for trial_id in trials
    }

    assert math.isnan(search_overhead(20.0, _spans(*intervals), by_trial))


def test_the_trial_spans_follow_ray_tune():
    """The driver-side callback records when each trial runs."""
    spans = TrialSpans()
    trial = SimpleNamespace(trial_id="a")

    spans.on_trial_start(iteration=0, trials=[], trial=trial)
    spans.on_trial_complete(iteration=1, trials=[], trial=trial)

    ((start, end),) = spans.spans["a"]
    assert 0.0 <= start <= end


def test_cross_validation_reports_the_trial_ids_of_its_best_configuration():
    """The folds of the winning configuration are the best trials."""
    frame = pd.DataFrame(
        {
            "trial_id": ["a0", "a1", "b0", "b1"],
            "config/k": [10, 10, 20, 20],
            "config/fold": [0, 1, 0, 1],
            "nDCG@10": [0.3, 0.32, 0.1, 0.12],
            "iterations": [1, 1, 1, 1],
        }
    )

    _, stats = trainer_module.Trainer(storage_path="unused")._aggregate_cv_results(
        frame, "nDCG@10", "max", "median"
    )

    assert sorted(stats["trial_ids"]) == ["a0", "a1"]


@pytest.mark.parametrize(
    "model_name, best_params",
    [
        ("BPR", {**BPR, "iterations": 2}),
        ("EASE", {"l2": 10.0, "iterations": 1}),
    ],
)
def test_the_cross_validation_retraining_reports_its_train_only_time(
    dataset: Dataset, model_name: str, best_params: Dict[str, Any]
):
    """The retraining on the full training set is timed as well."""
    params = params_registry.get(
        model_name,
        **{k: v for k, v in best_params.items() if k != "iterations"},
        optimization={"num_workers": 0},
    )

    _, report, _ = remote_model_retraining._function(  # type: ignore[attr-defined]
        model_name=model_name,
        best_params=best_params,
        main_dataset=dataset,
        params=params,
        custom_modules=[],
        device="cpu",
        seed=42,
    )

    assert report["Retraining Train-Only Time (s)"] > 0.0


def test_the_test_evaluation_and_inference_clocks_wait_for_the_device(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset
):
    """Both clocks start and stop only once the queued device work is done."""
    synchronised: List[str] = []
    monkeypatch.setattr(ml, "synchronize", synchronised.append)

    _, evaluation_s, inference_s = ml.remote_evaluation_and_timing._function(  # type: ignore[attr-defined]
        model=make_model("EASE", dataset),
        main_dataset=dataset,
        evaluation=EvaluationConfig(top_k=[5], metrics=["nDCG"]),
        num_workers=0,
        device="cpu",
        requires_timing=True,
        custom_modules=[],
    )

    assert synchronised == ["cpu"] * 4
    assert evaluation_s > 0.0
    assert inference_s > 0.0


def test_train_and_swarm_build_the_same_row():
    """The row the pipelines write keeps every existing column and adds the new."""
    row = model_timing_row(
        model_name="BPR",
        data_preparation_time=1.0,
        exploration_time=10.0,
        ray_report={"Average Trial Time (s)": 4.0, "Best Trial Epochs": 3},
        evaluation_time=2.0,
        inference_time=0.5,
    )

    # Without the data preparation's breakdown its columns are there, empty
    data_columns = DataPreparationTimes().columns()
    assert all(math.isnan(row.pop(column)) for column in data_columns)
    assert row == {
        "Model Name": "BPR",
        "Data Preparation Time": 1.0,
        "Hyperparameter Exploration Time": 10.0,
        "Average Trial Time (s)": 4.0,
        "Best Trial Epochs": 3,
        "Evaluation Time": 2.0,
        "Inference Time": 0.5,
        "Total Time": 13.0,
    }


def test_the_written_report_keeps_counts_and_rates_as_numbers(tmp_path: Path):
    """Durations, epochs and rates are all written as plain numbers."""
    writer = LocalWriter(dataset_name="timing", local_path=str(tmp_path))
    writer.write_time_report(
        [
            {
                "Model Name": "BPR",
                "Trainable Params (Best Model)": 0,
                "Total Params (Best Model)": 0,
                "Hyperparameter Exploration Time": 75.0,
                "Best Trial Epochs": 1.5,
                "Best Trial Train Throughput (samples/s)": 1234.5,
                "Best Trial Train-Only Time (s)": 4.25,
                "Inference Time": 0.0131,
            },
            {
                "Model Name": "EASE",
                "Trainable Params (Best Model)": 0,
                "Total Params (Best Model)": 0,
                "Hyperparameter Exploration Time": 3.0,
                "Best Trial Epochs": np.nan,
                "Best Trial Train Throughput (samples/s)": np.nan,
                "Best Trial Train-Only Time (s)": np.nan,
                "Inference Time": 0.0131,
            },
        ]
    )
    report = pd.read_csv(
        next(Path(writer.experiment_evaluation_path).glob("Time_Report_*.tsv")),
        sep="\t",
    ).set_index("Model Name")

    assert report.loc["BPR", "Best Trial Epochs"] == pytest.approx(1.5)
    assert report.loc["BPR", "Best Trial Train Throughput (samples/s)"] == (
        pytest.approx(1234.5)
    )
    assert report.loc["BPR", "Best Trial Train-Only Time (s)"] == pytest.approx(4.25)
    assert report.loc["BPR", "Hyperparameter Exploration Time"] == (pytest.approx(75.0))


def test_a_report_from_an_older_run_is_merged_in_seconds(tmp_path: Path):
    """A report holding h:mm:ss durations is read back as seconds and kept."""
    writer = LocalWriter(dataset_name="timing", local_path=str(tmp_path))
    evaluation = Path(writer.experiment_evaluation_path)
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / f"Time_Report_{writer.timestamp}.tsv").write_text(
        "Model Name\tTrainable Params (Best Model)\tTotal Params (Best Model)\t"
        "Data Preparation Time\tHyperparameter Exploration Time\t"
        "Average Trial Time (s)\tEvaluation Time\tTotal Time\tInference Time (ms)\n"
        "EASE\t0\t0\t0:00:16.953379\t1 day, 0:00:01.500000\t27.2\t"
        "0:00:00.111793\t1 day, 0:00:18.565172\t65.5\n"
    )

    writer.write_time_report(
        [
            {
                "Model Name": "BPR",
                "Trainable Params (Best Model)": 10,
                "Total Params (Best Model)": 10,
                "Data Preparation Time": 16.953379,
                "Hyperparameter Exploration Time": 75.0,
                "Average Trial Time (s)": 31.5,
                "Evaluation Time": 0.04,
                "Total Time": 91.993379,
                "Inference Time": 0.0429,
            }
        ]
    )
    report = pd.read_csv(
        next(evaluation.glob("Time_Report_*.tsv")), sep="\t"
    ).set_index("Model Name")

    assert report.loc["EASE", "Data Preparation Time"] == pytest.approx(16.953379)
    assert report.loc["EASE", "Hyperparameter Exploration Time"] == pytest.approx(
        86401.5
    )
    assert report.loc["EASE", "Evaluation Time"] == pytest.approx(0.111793)
    assert report.loc["EASE", "Inference Time (ms)"] == pytest.approx(65.5)
    assert report.loc["BPR", "Total Time"] == pytest.approx(91.993379)
    assert report.loc["BPR", "Inference Time (ms)"] == pytest.approx(42.9)
    for column in report.columns:
        assert pd.api.types.is_numeric_dtype(report[column]), column
