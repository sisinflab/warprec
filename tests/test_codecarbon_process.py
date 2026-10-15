"""CodeCarbon's 'process' tracking measures each Ray Train worker of a trial.

The CodeCarbon dashboard is a Ray Tune callback, and Ray Tune calls it on the
driver, so a trial measures itself where it trains. Its rows used to reach the
driver through files left under the driver's storage path, which a node
without that storage never saw: the rows were lost without a word. They now
travel through Ray, labelled with the node and the worker, and a worker
charges only the GPUs Ray assigned to it.
"""

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
import pytest

pytest.importorskip("codecarbon")

import ray  # noqa: E402

import warprec.recommenders  # noqa: E402,F401  (populates the registries)
from warprec.data.dataset import Dataset  # noqa: E402
from warprec.recommenders.trainer import emissions as emissions_module  # noqa: E402
from warprec.recommenders.trainer import trainer as trainer_module  # noqa: E402
from warprec.recommenders.trainer.dashboard_callbacks import (  # noqa: E402
    CodeCarbonCallback,
)
from warprec.recommenders.trainer.emissions import (  # noqa: E402
    EmissionsRecorder,
    EmissionsSettings,
    trial_emissions,
)
from warprec.utils.config import DashboardConfig  # noqa: E402
from warprec.utils.config.evaluation_configuration import (  # noqa: E402
    EvaluationConfig,
)
from warprec.utils.registry import params_registry  # noqa: E402


@pytest.fixture(scope="module")
def local_ray() -> Iterator[None]:
    """A local Ray instance for the tests that need one.

    Yields:
        None: While the tests run.
    """
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=2, include_dashboard=False)
    yield
    if started:
        ray.shutdown()


@ray.remote(num_cpus=0)
def train_a_trial(
    settings: EmissionsSettings, trial_id: str, seconds: float, rank: int = 0
) -> None:
    """What a Ray Train worker does around its training, in a process of its own.

    Args:
        settings (EmissionsSettings): What the trial received from the driver.
        trial_id (str): The trial.
        seconds (float): How long to train.
        rank (int): The worker's rank in the trial.
    """
    emissions_module.world_rank = lambda: rank
    with trial_emissions(settings, trial_id):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            sum(i * i for i in range(10_000))


def labels(rows: pd.DataFrame) -> List[Dict[str, str]]:
    """The labels of the rows of emissions.csv.

    Args:
        rows (pd.DataFrame): The rows.

    Returns:
        List[Dict[str, str]]: Each row's label, as key-value pairs.
    """
    return [
        dict(part.split("=", 1) for part in label.split())
        for label in rows["experiment_id"]
    ]


def attentions(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Record what is logged as needing the user's attention.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the logger's method.

    Returns:
        List[str]: The messages, filled as they are logged.
    """
    messages: List[str] = []
    monkeypatch.setattr(
        emissions_module.logger,
        "attention",
        lambda message, *a, **k: messages.append(message),
    )
    return messages


def process_recorder(tmp_path: Path) -> EmissionsRecorder:
    """A process-mode recorder writing to a temporary folder.

    Args:
        tmp_path (Path): Where emissions.csv is written.

    Returns:
        EmissionsRecorder: The recorder, not started.
    """
    return EmissionsRecorder(
        tracking_mode="process",
        save_to_api=False,
        save_to_file=True,
        output_dir=str(tmp_path),
        project_name="run-x",
        poll_interval=0.5,
    )


def _trial_bundle(
    monkeypatch: pytest.MonkeyPatch,
    dataset: Dataset,
    tmp_path: Path,
    emissions: EmissionsSettings,
) -> Dict[str, Any]:
    """The data a train-pipeline trial receives, with CodeCarbon enabled.

    Args:
        monkeypatch (pytest.MonkeyPatch): Keeps Ray out of the test.
        dataset (Dataset): The dataset to optimise on.
        tmp_path (Path): Where Ray Tune would store its results.
        emissions (EmissionsSettings): The measurement of the run the search
            belongs to.

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
            "tracking_mode": emissions.tracking_mode,
        }
    )
    trainer_module.Trainer(
        storage_path=str(tmp_path), dashboard_config=dashboard, emissions=emissions
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
    inbox = object()  # Stands in for the run's actor
    process = EmissionsSettings("process", False, "run-x", inbox=inbox)
    machine = EmissionsSettings("machine", False, "run-x")

    assert _trial_bundle(monkeypatch, dataset, tmp_path, process)["codecarbon"] == (
        process
    )
    assert _trial_bundle(monkeypatch, dataset, tmp_path, machine)["codecarbon"] is None


def test_the_rows_of_the_workers_reach_the_driver_through_ray(
    local_ray: None, tmp_path: Path
):
    """Each worker's row arrives through Ray, labelled, with no file on the way."""
    recorder = process_recorder(tmp_path)
    with recorder as settings:
        ray.get(
            [train_a_trial.remote(settings, "abc_00000", 0.2, rank) for rank in (0, 1)]
            + [train_a_trial.remote(settings, "abc_00001", 0.2)]
        )

    rows = pd.read_csv(tmp_path / "emissions.csv")
    found = sorted((label["trial"], label["rank"]) for label in labels(rows))
    assert found == [("abc_00000", "0"), ("abc_00000", "1"), ("abc_00001", "0")]
    node = ray.get_runtime_context().get_node_id()
    assert {label["node"] for label in labels(rows)} == {node}
    assert (rows["project_name"] == "run-x").all()
    assert (rows["tracking_mode"] == "process").all()
    # Nothing is staged on the way: the driver's file is all there is
    assert [path.name for path in tmp_path.iterdir()] == ["emissions.csv"]


def test_the_trial_row_carries_the_utilization_sampled_while_it_trained(
    local_ray: None, tmp_path: Path
):
    """The row used to read 0 for the CPU and RAM utilization and the RAM used.

    CodeCarbon samples them every second only when its tracker has an output to
    write to, and the trial's tracker writes nowhere: the driver writes the row.
    The RAM is read from the first sample on; the CPU utilization of CodeCarbon's
    first two samples is 0 (psutil measures it since the previous call from the
    same thread, and each sample runs in a new one), so it is not asserted on a
    trial this short.
    """
    recorder = process_recorder(tmp_path)
    with recorder as settings:
        ray.get(train_a_trial.remote(settings, "abc_00000", 2.5))

    row = pd.read_csv(tmp_path / "emissions.csv").iloc[0]
    assert row["ram_utilization_percent"] > 0
    assert row["ram_used_gb"] > 0


def test_a_finished_trial_whose_rows_did_not_arrive_is_logged(
    local_ray: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    recorder = process_recorder(tmp_path)
    with recorder as settings:
        callback = CodeCarbonCallback(
            save_to_file=True,
            output_dir=str(tmp_path),
            tracking_mode="process",
            settings=settings,
        )
        assert callback.trial_settings(num_workers=2) == settings

        # Both workers of the first trial deliver, one of the second's is lost
        ray.get(
            [train_a_trial.remote(settings, "abc_00000", 0.1, rank) for rank in (0, 1)]
            + [train_a_trial.remote(settings, "abc_00001", 0.1)]
        )
        for trial_id in ("abc_00000", "abc_00001"):
            trial = SimpleNamespace(trial_id=trial_id)
            callback.on_trial_complete(iteration=1, trials=[trial], trial=trial)

    assert len(messages) == 1
    assert "abc_00001" in messages[0]


def test_a_search_on_its_own_collects_its_trials_rows(local_ray: None, tmp_path: Path):
    """Without a run around it, the callback collects the rows itself."""
    callback = CodeCarbonCallback(
        save_to_file=True, output_dir=str(tmp_path), tracking_mode="process"
    )
    settings = callback.trial_settings(num_workers=1)
    assert settings is not None
    ray.get(train_a_trial.remote(settings, "abc_00000", 0.1))
    trial = SimpleNamespace(trial_id="abc_00000")
    callback.on_trial_complete(iteration=1, trials=[trial], trial=trial)
    callback.on_experiment_end(trials=[trial])

    rows = pd.read_csv(tmp_path / "emissions.csv")
    assert [label["trial"] for label in labels(rows)] == ["abc_00000"]


class _Tracker:
    """Stands in for CodeCarbon's tracker, keeping what it was built with.

    Args:
        **kwargs (Any): The tracker's arguments.

    Attributes:
        built (List[Dict[str, Any]]): The arguments of every tracker built.
    """

    built: List[Dict[str, Any]] = []

    def __init__(self, **kwargs: Any):
        _Tracker.built.append(kwargs)

    def start(self) -> None:
        """Start measuring."""

    def stop(self) -> Optional[float]:
        """Stop measuring.

        Returns:
            Optional[float]: Nothing measured.
        """
        return None


@pytest.mark.parametrize("assigned", [["1"], [0, 2], []])
def test_a_worker_measures_only_the_gpus_ray_assigned_to_it(
    monkeypatch: pytest.MonkeyPatch, assigned: List[Any]
):
    """Ray Train shares CUDA_VISIBLE_DEVICES among the workers of a node, and
    CodeCarbon measures every GPU listed there, or every GPU of the node when
    it is unset: each worker was charged for its neighbours' GPUs. A worker
    without a GPU is charged for none."""
    _Tracker.built = []
    monkeypatch.setattr(emissions_module, "EmissionsTracker", _Tracker)
    monkeypatch.setattr(emissions_module.ray, "get_gpu_ids", lambda: assigned)

    with trial_emissions(EmissionsSettings("process", False, "run-x"), "abc_00000"):
        pass

    assert [tracker["gpu_ids"] for tracker in _Tracker.built] == [assigned]
    assert _Tracker.built[0]["tracking_mode"] == "process"
