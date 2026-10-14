"""CodeCarbon's 'machine' tracking measures every Ray node for the whole run.

The trackers used to start on the driver when a trial started: every row was
the head node's, the other nodes were never measured, concurrent trials each
counted the whole head node, and only the searches were covered. A run now
keeps one tracker on each node, from the moment its work starts to its end,
and the driver writes the rows the trackers send back through Ray.
"""

import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List

import pandas as pd
import pytest

pytest.importorskip("codecarbon")

import ray  # noqa: E402
from ray.cluster_utils import Cluster  # noqa: E402

from warprec.pipelines import swarm as swarm_module  # noqa: E402
from warprec.pipelines import train as train_module  # noqa: E402
from warprec.pipelines.common import measure_emissions  # noqa: E402
from warprec.recommenders.trainer import emissions as emissions_module  # noqa: E402
from warprec.recommenders.trainer.dashboard_callbacks import (  # noqa: E402
    CodeCarbonCallback,
)
from warprec.recommenders.trainer.emissions import EmissionsRecorder  # noqa: E402
from warprec.utils.config import DashboardConfig  # noqa: E402


@pytest.fixture(scope="module")
def cluster() -> Iterator[Cluster]:
    """A two-node Ray cluster on this machine, connected to.

    Yields:
        Cluster: The cluster, shut down at the end of the module.
    """
    if ray.is_initialized():
        ray.shutdown()
    local = Cluster(
        initialize_head=True,
        head_node_args={"num_cpus": 1, "include_dashboard": False},
    )
    local.add_node(num_cpus=1)
    local.wait_for_nodes()
    ray.init(address=local.address)
    yield local
    ray.shutdown()
    local.shutdown()


def alive_nodes() -> List[str]:
    """The ids of the nodes of the cluster that are alive.

    Returns:
        List[str]: The node ids.
    """
    return [node["NodeID"] for node in ray.nodes() if node["Alive"]]


def measured_nodes(rows: pd.DataFrame) -> List[str]:
    """The node each row was measured on, read from its label.

    Args:
        rows (pd.DataFrame): The rows of emissions.csv.

    Returns:
        List[str]: The node id of each row.
    """
    return [
        dict(part.split("=", 1) for part in label.split())["node"]
        for label in rows["experiment_id"]
    ]


def wait_for(condition: Callable[[], bool], timeout: float = 60.0) -> None:
    """Wait until a condition holds.

    Args:
        condition (Callable[[], bool]): What to wait for.
        timeout (float): How long to wait, in seconds.

    Raises:
        TimeoutError: If the condition never held.
    """
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > end:
            raise TimeoutError("The condition never held.")
        time.sleep(0.2)


def attentions(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Record what the recorder logs as needing the user's attention.

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


def recorder(tmp_path: Path, **kwargs: Any) -> EmissionsRecorder:
    """A machine-mode recorder writing to a temporary folder.

    Args:
        tmp_path (Path): Where emissions.csv is written.
        **kwargs (Any): Overrides of the recorder's arguments.

    Returns:
        EmissionsRecorder: The recorder, not started.
    """
    arguments: Dict[str, Any] = {
        "tracking_mode": "machine",
        "save_to_api": False,
        "save_to_file": True,
        "output_dir": str(tmp_path),
        "project_name": "run-x",
        "poll_interval": 0.5,
    }
    arguments.update(kwargs)
    return EmissionsRecorder(**arguments)


def test_every_node_gets_one_row_for_the_whole_run(cluster: Cluster, tmp_path: Path):
    started = time.monotonic()
    with recorder(tmp_path):
        time.sleep(3)
    elapsed = time.monotonic() - started

    rows = pd.read_csv(tmp_path / "emissions.csv")
    assert sorted(measured_nodes(rows)) == sorted(alive_nodes())
    assert rows["project_name"].tolist() == ["run-x", "run-x"]
    assert (rows["tracking_mode"] == "machine").all()
    # The trackers start alongside the run, a few seconds into it, and
    # measure it to its end
    assert (rows["duration"] > 0).all()
    assert (rows["duration"] <= elapsed).all()


def test_a_node_that_joins_during_the_run_is_measured(cluster: Cluster, tmp_path: Path):
    running = recorder(tmp_path)
    with running:
        joined = cluster.add_node(num_cpus=1)
        cluster.wait_for_nodes()
        try:
            wait_for(lambda: len(running.nodes) == 3)
            time.sleep(1)
        finally:
            running_nodes = list(running.nodes)
    rows = pd.read_csv(tmp_path / "emissions.csv")
    cluster.remove_node(joined)

    assert len(running_nodes) == 3
    assert sorted(measured_nodes(rows)) == sorted(running_nodes)


def test_a_node_that_leaves_loses_its_row_and_says_so(
    cluster: Cluster, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    leaving = cluster.add_node(num_cpus=1)
    cluster.wait_for_nodes()
    running = recorder(tmp_path)
    with running:
        wait_for(lambda: len(running.nodes) == 3)
        left = leaving.node_id
        cluster.remove_node(leaving)
        wait_for(lambda: left not in alive_nodes())
    rows = pd.read_csv(tmp_path / "emissions.csv")

    assert left not in measured_nodes(rows)
    assert sorted(measured_nodes(rows)) == sorted(alive_nodes())
    assert any(left in message for message in messages)


def test_the_run_level_trackers_replace_the_per_trial_ones(
    cluster: Cluster, tmp_path: Path
):
    """A trial starting no longer starts a tracker on the driver."""
    running = recorder(tmp_path)
    with running:
        callback = CodeCarbonCallback(
            save_to_file=True,
            output_dir=str(tmp_path),
            tracking_mode="machine",
            settings=running.settings,
        )
        trial = SimpleNamespace(trial_id="abc_00000")
        assert callback.trial_settings(num_workers=1) is None
        callback.on_trial_start(iteration=0, trials=[trial], trial=trial)
        callback.on_trial_complete(iteration=1, trials=[trial], trial=trial)
        callback.on_experiment_end(trials=[trial])
        assert not (tmp_path / "emissions.csv").exists()

    rows = pd.read_csv(tmp_path / "emissions.csv")
    assert sorted(measured_nodes(rows)) == sorted(alive_nodes())


def test_a_trainer_used_on_its_own_measures_the_nodes_during_its_search(
    cluster: Cluster, tmp_path: Path
):
    """Without a run around it, the callback measures the nodes itself."""
    callback = CodeCarbonCallback(
        save_to_file=True, output_dir=str(tmp_path), tracking_mode="machine"
    )
    assert callback.trial_settings(num_workers=1) is None
    time.sleep(1)
    callback.on_experiment_end(trials=[])

    rows = pd.read_csv(tmp_path / "emissions.csv")
    assert sorted(measured_nodes(rows)) == sorted(alive_nodes())


@pytest.mark.parametrize("pipeline", ["train", "swarm"])
def test_the_pipelines_measure_from_data_preparation_and_stop_on_failure(
    monkeypatch: pytest.MonkeyPatch, pipeline: str
):
    module = train_module if pipeline == "train" else swarm_module
    events: List[str] = []
    context = SimpleNamespace(config=SimpleNamespace(), run_name="run-x")

    @contextmanager
    def measured(ctx: Any) -> Iterator[None]:
        """Stand in for the run's measurement.

        Args:
            ctx (Any): The pipeline context.

        Yields:
            None: While the run works.
        """
        assert ctx is context
        events.append("start")
        try:
            yield None
        finally:
            events.append("stop")

    def prepare(ctx: Any) -> None:
        """Fail the data preparation.

        Args:
            ctx (Any): The pipeline context.

        Raises:
            RuntimeError: Always.
        """
        events.append("prepare")
        raise RuntimeError("boom")

    monkeypatch.setattr(
        module,
        "bootstrap_pipeline",
        lambda *a, **k: events.append("bootstrap") or context,
    )
    monkeypatch.setattr(module, "measure_emissions", measured)
    monkeypatch.setattr(module, "prepare_datasets", prepare)

    entry = module.train_pipeline if pipeline == "train" else module.swarm_pipeline
    with pytest.raises(RuntimeError, match="boom"):
        entry("unused.yml")
    assert events == ["bootstrap", "start", "prepare", "stop"]


def _context(tmp_path: Path, **codecarbon: Any) -> SimpleNamespace:
    """A pipeline context with a CodeCarbon section.

    Args:
        tmp_path (Path): The storage path of the run.
        **codecarbon (Any): The CodeCarbon settings.

    Returns:
        SimpleNamespace: The context 'measure_emissions' reads.
    """
    config = SimpleNamespace(
        dashboard=DashboardConfig(codecarbon=codecarbon),
        get_storage_path=lambda: str(tmp_path),
    )
    return SimpleNamespace(config=config, run_name="run-x")


def test_a_run_without_codecarbon_measures_nothing(tmp_path: Path):
    with measure_emissions(_context(tmp_path)) as settings:
        assert settings is None


def test_a_run_with_codecarbon_records_into_its_storage_path(tmp_path: Path):
    measurement = measure_emissions(
        _context(tmp_path, enabled=True, save_to_file=True, tracking_mode="machine")
    )
    assert isinstance(measurement, EmissionsRecorder)
    assert measurement.output_dir == str(tmp_path)
    assert measurement.project_name == "run-x"
    assert measurement.tracking_mode == "machine"
    assert measurement.save_to_file
