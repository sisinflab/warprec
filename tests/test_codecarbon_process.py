"""CodeCarbon's 'process' tracking measures the process a trial trains in.

The CodeCarbon dashboard is a Ray Tune callback, and Ray Tune calls it on the
driver. Its trackers were started there too, so 'tracking_mode: process'
measured the driver, which waits for the trials, rather than the Ray Train
worker that trains. With 'process' tracking the trial now measures itself and
the driver only writes the rows to 'emissions.csv'.
"""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pandas as pd
import pytest

pytest.importorskip("codecarbon")

import warprec.recommenders  # noqa: E402,F401  (populates the registries)
from warprec.data.dataset import Dataset  # noqa: E402
from warprec.recommenders.trainer import trainer as trainer_module  # noqa: E402
from warprec.recommenders.trainer.dashboard_callbacks import (  # noqa: E402
    CodeCarbonCallback,
    trial_emissions,
)
from warprec.utils.config import DashboardConfig  # noqa: E402
from warprec.utils.config.evaluation_configuration import (  # noqa: E402
    EvaluationConfig,
)
from warprec.utils.registry import params_registry  # noqa: E402


def _trial_bundle(
    monkeypatch: pytest.MonkeyPatch,
    dataset: Dataset,
    tmp_path: Path,
    tracking_mode: str,
) -> Dict[str, Any]:
    """The data a train-pipeline trial receives, with CodeCarbon enabled.

    Args:
        monkeypatch (pytest.MonkeyPatch): Keeps Ray out of the test.
        dataset (Dataset): The dataset to optimise on.
        tmp_path (Path): Where Ray Tune would store its results.
        tracking_mode (str): CodeCarbon's tracking mode.

    Returns:
        Dict[str, Any]: The data bundle of the trials.
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
        "BPR",
        embedding_size=8,
        reg_weight=1e-5,
        batch_size=64,
        epochs=1,
        learning_rate=0.001,
        optimization={"max_concurrent_trials": 1},
    )
    dashboard = DashboardConfig(
        codecarbon={
            "enabled": True,
            "save_to_file": True,
            "tracking_mode": tracking_mode,
        }
    )
    trainer_module.Trainer(
        storage_path=str(tmp_path), dashboard_config=dashboard
    )._setup_tuner(
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
    return captured["data_bundle"]


def test_process_tracking_travels_to_the_trial(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset, tmp_path: Path
):
    """The trials are told to measure themselves under 'process' tracking only."""
    process = _trial_bundle(monkeypatch, dataset, tmp_path, "process")
    machine = _trial_bundle(monkeypatch, dataset, tmp_path, "machine")

    assert process["codecarbon"] == {
        "save_to_api": False,
        "save_to_file": True,
        "output_dir": str(tmp_path),
    }
    assert machine["codecarbon"] is None


def test_the_driver_writes_the_row_the_trial_measured(tmp_path: Path):
    """The driver starts no tracker and writes the trial's own measurement."""
    callback = CodeCarbonCallback(
        save_to_file=True, output_dir=str(tmp_path), tracking_mode="process"
    )
    trial = SimpleNamespace(trial_id="abc_00000")

    callback.on_trial_start(iteration=0, trials=[trial], trial=trial)
    assert not callback.trackers

    # What the trial process does around its training
    with trial_emissions(callback.trial_settings(), trial.trial_id):
        sum(i * i for i in range(200_000))

    callback.on_trial_complete(iteration=1, trials=[trial], trial=trial)

    rows = pd.read_csv(tmp_path / "emissions.csv")
    assert rows["project_name"].tolist() == ["abc_00000"]
    assert rows["duration"].iloc[0] > 0


def test_the_trial_row_carries_the_utilization_sampled_while_it_trained(
    tmp_path: Path,
):
    """The row used to read 0 for the CPU and RAM utilization and the RAM used.

    CodeCarbon samples them every second only when its tracker has an output to
    write to, and the trial's tracker wrote nowhere: the driver writes the row.
    The RAM is read from the first sample on; the CPU utilization of CodeCarbon's
    first two samples is 0 (psutil measures it since the previous call from the
    same thread, and each sample runs in a new one), so it is not asserted on a
    trial this short.
    """
    callback = CodeCarbonCallback(
        save_to_file=True, output_dir=str(tmp_path), tracking_mode="process"
    )
    trial = SimpleNamespace(trial_id="abc_00000")

    with trial_emissions(callback.trial_settings(), trial.trial_id):
        end = time.perf_counter() + 2.5
        while time.perf_counter() < end:
            sum(i * i for i in range(10_000))

    callback.on_trial_complete(iteration=1, trials=[trial], trial=trial)

    row = pd.read_csv(tmp_path / "emissions.csv").iloc[0]
    assert row["tracking_mode"] == "process"
    assert row["ram_utilization_percent"] > 0
    assert row["ram_used_gb"] > 0
