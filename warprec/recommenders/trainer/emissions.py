import os
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

import ray
from codecarbon import EmissionsTracker
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData
from codecarbon.output_methods.file import FileOutput
from ray.exceptions import GetTimeoutError, RayError
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from warprec.utils.logger import logger

EMISSIONS_FILE = "emissions.csv"

# How often a run looks for nodes that joined the cluster, in seconds
NODE_POLL_SECONDS = 10.0

# How long the driver waits for a tracker to hand its row back, in seconds
TRACKER_TIMEOUT_SECONDS = 120.0

# How long a run waits for its trackers to start measuring, in seconds
TRACKER_START_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class EmissionsSettings:
    """What the trainers of a run need to know about its CodeCarbon measurement.

    It travels to the Ray Tune driver of every model, a Ray task in the swarm
    pipeline, so it holds only what can be sent there.

    Attributes:
        tracking_mode (str): CodeCarbon's tracking mode, 'machine' or 'process'.
        save_to_api (bool): Whether the trackers send their emissions to the
            CodeCarbon API.
        project_name (str): The project name of every row, the run's name.
        inbox (Any): Under 'process' tracking, the 'EmissionsInbox' actor the
            trial workers hand their rows to. None otherwise.
        startup_seconds (float): How long the run waited for its trackers to
            start, which its data preparation time leaves out.
    """

    tracking_mode: str
    save_to_api: bool
    project_name: str
    inbox: Any = None
    startup_seconds: float = 0.0


class _RowCapture(BaseOutput):
    """Keep the row a tracker measured, for the driver to write.

    CodeCarbon samples the CPU and RAM utilization and the RAM used every
    second, and averages them into the row, only while its tracker has an
    output to write to. The trackers write nowhere themselves, so without one
    the row read 0 for all three.

    Attributes:
        total (Optional[EmissionsData]): The row, once the tracker stopped.
    """

    total: Optional[EmissionsData]

    def __init__(self) -> None:
        self.total = None

    def out(self, total: EmissionsData, delta: EmissionsData) -> None:
        """Keep the row the tracker writes when it stops.

        Args:
            total (EmissionsData): The emissions since the tracker started.
            delta (EmissionsData): The emissions since the last write.
        """
        self.total = total


def node_label() -> str:
    """The label of the Ray node the calling process runs on.

    Returns:
        str: The node's host name, IP address and Ray node id, as
            'host=<name> ip=<address> node=<id>'.
    """
    return (
        f"host={socket.gethostname()} ip={ray.util.get_node_ip_address()} "
        f"node={ray.get_runtime_context().get_node_id()}"
    )


def labelled_row(row: EmissionsData, label: str) -> Dict[str, Any]:
    """The row to write to 'emissions.csv', labelled with what it measured.

    The label goes in 'experiment_id', so that the file keeps CodeCarbon's
    columns. Only the row written to the file is relabelled: what a tracker
    sends to the CodeCarbon API keeps the API's experiment.

    Args:
        row (EmissionsData): The row a tracker measured.
        label (str): What the row measured.

    Returns:
        Dict[str, Any]: The fields of the row.
    """
    data = asdict(row)
    data["experiment_id"] = label
    return data


@ray.remote(num_cpus=0)
class NodeEmissionsTracker:  # pylint: disable = too-few-public-methods
    """A CodeCarbon 'machine' tracker kept on one Ray node.

    Args:
        save_to_api (bool): Whether to send the emissions to the CodeCarbon API.
        project_name (str): The project name of the row.
    """

    def __init__(self, save_to_api: bool, project_name: str):
        self._row = _RowCapture()
        self._label = node_label()
        self._tracker = EmissionsTracker(
            project_name=project_name,
            save_to_api=save_to_api,
            save_to_file=False,
            tracking_mode="machine",
            log_level="error",
            output_handlers=[self._row],
        )
        self._tracker.start()

    def ready(self) -> bool:
        """Answer once the tracker measures, for the run to wait on.

        Returns:
            bool: Always True.
        """
        return True

    def stop(self) -> Optional[Dict[str, Any]]:
        """Stop measuring and hand back the row.

        Returns:
            Optional[Dict[str, Any]]: The labelled row, or None when CodeCarbon
                produced none.
        """
        self._tracker.stop()
        if self._row.total is None:
            return None
        return labelled_row(self._row.total, self._label)


@ray.remote(num_cpus=0)
class EmissionsInbox:
    """Where the trial workers of a run hand their rows to the driver.

    The rows travel through Ray, so a worker on a node that cannot see the
    driver's storage delivers its row all the same.
    """

    def __init__(self) -> None:
        self._rows: List[Dict[str, Any]] = []
        self._delivered: Dict[str, int] = {}

    def put(self, trial_id: str, row: Dict[str, Any]) -> None:
        """Keep the row a worker of a trial measured.

        Args:
            trial_id (str): The trial.
            row (Dict[str, Any]): The labelled row.
        """
        self._rows.append(row)
        self._delivered[trial_id] = self._delivered.get(trial_id, 0) + 1

    def take(self) -> List[Dict[str, Any]]:
        """Hand over the rows kept since the last call.

        Returns:
            List[Dict[str, Any]]: The rows.
        """
        rows, self._rows = self._rows, []
        return rows

    def delivered(self, trial_id: str) -> int:
        """How many rows the workers of a trial delivered.

        Args:
            trial_id (str): The trial.

        Returns:
            int: The number of rows.
        """
        return self._delivered.get(trial_id, 0)


def assigned_gpu_ids() -> List[Union[int, str]]:
    """The GPUs Ray assigned to the calling worker, for CodeCarbon to measure.

    Left to itself, CodeCarbon measures every GPU listed in
    'CUDA_VISIBLE_DEVICES', which Ray Train shares among the workers of a
    node, or every GPU of the node when it is unset: each worker would be
    charged for its neighbours' GPUs. An empty list measures none, so a worker
    without a GPU is charged for none.

    Returns:
        List[Union[int, str]]: The ids of the GPUs.
    """
    return list(ray.get_gpu_ids())


def world_rank() -> int:
    """The rank of the calling worker among the Ray Train workers of its trial.

    Returns:
        int: The rank, or 0 outside a Ray Train worker.
    """
    try:
        return ray.train.get_context().get_world_rank()
    except RuntimeError:
        return 0


@contextmanager
def trial_emissions(
    settings: Optional[EmissionsSettings], trial_id: Optional[str]
) -> Iterator[None]:
    """Measure the emissions of the Ray Train worker a trial trains in.

    CodeCarbon's 'process' mode measures the process that starts the tracker,
    and the driver's Ray Tune callback runs in the driver, not where the trial
    trains. With 'process' tracking each worker of the trial therefore
    measures itself, only on the GPUs Ray assigned to it. It sends its
    emissions to the CodeCarbon API itself, when asked to, and hands its row,
    labelled with its node, trial and rank, to the run's inbox, through Ray,
    for the driver to write.

    Args:
        settings (Optional[EmissionsSettings]): The run's measurement, or None
            when the trial is not measured.
        trial_id (Optional[str]): The Ray Tune trial.

    Yields:
        None: While the trial trains.
    """
    if settings is None:
        yield
        return

    row = _RowCapture()
    tracker = EmissionsTracker(
        project_name=settings.project_name,
        save_to_api=settings.save_to_api,
        save_to_file=False,
        tracking_mode="process",
        log_level="error",
        output_handlers=[row],
        gpu_ids=assigned_gpu_ids(),
    )
    tracker.start()
    try:
        yield
    finally:
        tracker.stop()
        if row.total is not None and settings.inbox is not None:
            label = f"{node_label()} trial={trial_id} rank={world_rank()}"
            try:
                ray.get(
                    settings.inbox.put.remote(trial_id, labelled_row(row.total, label)),
                    timeout=TRACKER_TIMEOUT_SECONDS,
                )
            except (RayError, GetTimeoutError) as error:
                logger.attention(
                    f"The CodeCarbon row of trial {trial_id} ({label}) could not "
                    f"reach the driver: {type(error).__name__}."
                )


class EmissionsRecorder:
    """The CodeCarbon measurement of a whole run, held by its driver.

    Under 'machine' tracking it keeps one tracker on every Ray node, an actor
    pinned to the node that reserves no CPU, from 'start' to 'stop'. Nodes
    that join the cluster in between are looked for every 'poll_interval'
    seconds and get a tracker too. 'start' returns once every tracker measures,
    so the run's work is measured from its first step; a tracker that does not
    start within 'TRACKER_START_TIMEOUT_SECONDS' is logged and measures its
    node from whenever it starts. On 'stop' every tracker hands its row back
    through Ray and the driver appends them to 'emissions.csv', one row per
    node; a node that left the cluster before then has lost its row.

    Under 'process' tracking the trial workers measure themselves (see
    'trial_emissions') and hand their rows to an 'EmissionsInbox' actor kept
    on the driver's node, which the driver empties into 'emissions.csv' every
    'poll_interval' seconds and on 'stop'.

    Args:
        tracking_mode (str): CodeCarbon's tracking mode, 'machine' or 'process'.
        save_to_api (bool): Whether the trackers send their emissions to the
            CodeCarbon API.
        save_to_file (bool): Whether the driver writes 'emissions.csv'.
        output_dir (str): The folder of 'emissions.csv'.
        project_name (str): The project name of every row, the run's name.
        poll_interval (float): How often to look for nodes that joined, or for
            rows the trial workers delivered, in seconds.
    """

    def __init__(
        self,
        tracking_mode: str,
        save_to_api: bool,
        save_to_file: bool,
        output_dir: str,
        project_name: str,
        poll_interval: float = NODE_POLL_SECONDS,
    ):
        self.tracking_mode = tracking_mode
        self.save_to_api = save_to_api
        self.save_to_file = save_to_file
        self.output_dir = output_dir
        self.project_name = project_name
        self.poll_interval = poll_interval
        self._trackers: Dict[str, Any] = {}
        self._hosts: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._poller: Optional[threading.Thread] = None
        self._poll_error: Optional[str] = None
        self._inbox: Any = None
        self._startup_seconds = 0.0

    @property
    def settings(self) -> EmissionsSettings:
        """What the trainers of the run need to know about the measurement.

        Returns:
            EmissionsSettings: The settings of the run's measurement.
        """
        return EmissionsSettings(
            tracking_mode=self.tracking_mode,
            save_to_api=self.save_to_api,
            project_name=self.project_name,
            inbox=self._inbox,
            startup_seconds=self._startup_seconds,
        )

    @property
    def nodes(self) -> List[str]:
        """The Ray nodes measured so far.

        Returns:
            List[str]: Their node ids.
        """
        with self._lock:
            return list(self._trackers)

    def __enter__(self) -> EmissionsSettings:
        self.start()
        return self.settings

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    def start(self) -> None:
        """Start measuring, returning once every tracker measures."""
        started = time.monotonic()
        if self.tracking_mode == "process":
            self._inbox = EmissionsInbox.options(  # type: ignore[attr-defined]
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                ),
            ).remote()
            # The trial workers hand their rows to it from the first trial
            ray.get(self._inbox.take.remote(), timeout=TRACKER_START_TIMEOUT_SECONDS)
        else:
            self._wait_until_measuring(self._track_new_nodes())
        self._startup_seconds = time.monotonic() - started
        logger.msg(
            f"CodeCarbon is measuring the run ({self.tracking_mode} tracking), "
            f"started in {self._startup_seconds:.1f} seconds."
        )
        self._stopping.clear()
        self._poller = threading.Thread(
            target=self._poll, name="warprec-codecarbon", daemon=True
        )
        self._poller.start()

    def stop(self) -> None:
        """Stop measuring and write the rows the trackers hand back."""
        self._stopping.set()
        if self._poller is not None:
            self._poller.join()
            self._poller = None

        with self._lock:
            trackers, self._trackers = self._trackers, {}
        pending = {
            node_id: tracker.stop.remote() for node_id, tracker in trackers.items()
        }
        rows = []
        for node_id, reference in pending.items():
            try:
                row = ray.get(reference, timeout=TRACKER_TIMEOUT_SECONDS)
            except (RayError, GetTimeoutError) as error:
                logger.attention(
                    f"CodeCarbon could not measure node {self._hosts[node_id]} "
                    f"(node={node_id}) to the end of the run, and emissions.csv "
                    f"has no row for it: {type(error).__name__}."
                )
                continue
            if row is not None:
                rows.append(row)
        for tracker in trackers.values():
            ray.kill(tracker)
        self._write(rows)

        if self._inbox is not None:
            self._collect()
            ray.kill(self._inbox)
            self._inbox = None

    def _poll(self) -> None:
        """Look for nodes that joined, or for delivered rows, until the run stops."""
        while not self._stopping.wait(self.poll_interval):
            try:
                if self._inbox is not None:
                    self._collect()
                else:
                    self._track_new_nodes()
            except Exception as error:  # pylint: disable = broad-exception-caught
                # The thread must outlive a failed look, logged once: the next
                # look may well succeed
                if self._poll_error is None:
                    self._poll_error = str(error)
                    logger.attention(
                        f"CodeCarbon's measurement of the run failed a check: {error}"
                    )

    def _collect(self) -> None:
        """Write the rows the trial workers delivered since the last look."""
        self._write(ray.get(self._inbox.take.remote()))

    def _wait_until_measuring(self, trackers: Dict[str, Any]) -> None:
        """Wait until the given trackers measure, or the start timeout passes.

        Args:
            trackers (Dict[str, Any]): The trackers just started, by node id.
        """
        if not trackers:
            return
        answers = {tracker.ready.remote(): node for node, tracker in trackers.items()}
        ready, _ = ray.wait(
            list(answers),
            num_returns=len(answers),
            timeout=TRACKER_START_TIMEOUT_SECONDS,
        )
        for answer, node_id in answers.items():
            if answer not in ready:
                logger.attention(
                    f"CodeCarbon's tracker on node {self._hosts[node_id]} "
                    f"(node={node_id}) did not start within "
                    f"{TRACKER_START_TIMEOUT_SECONDS:.0f} seconds; the run goes on "
                    "and the node is measured from whenever it starts."
                )

    def _track_new_nodes(self) -> Dict[str, Any]:
        """Start a tracker on every alive node that has none.

        Returns:
            Dict[str, Any]: The trackers started, by node id.
        """
        started: Dict[str, Any] = {}
        for node in ray.nodes():
            node_id = node["NodeID"]
            if not node["Alive"] or node_id in self._trackers:
                continue
            tracker = NodeEmissionsTracker.options(  # type: ignore[attr-defined]
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
            ).remote(self.save_to_api, self.project_name)
            with self._lock:
                self._trackers[node_id] = tracker
                self._hosts[node_id] = (
                    f"{node.get('NodeManagerHostname')} "
                    f"({node.get('NodeManagerAddress')})"
                )
            started[node_id] = tracker
        return started

    def _write(self, rows: List[Dict[str, Any]]) -> None:
        """Append rows to 'emissions.csv', when the run saves to a file.

        Args:
            rows (List[Dict[str, Any]]): The labelled rows.
        """
        if not self.save_to_file or not rows:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        output = FileOutput(EMISSIONS_FILE, self.output_dir, on_csv_write="append")
        for row in rows:
            data = EmissionsData(**row)
            output.out(data, data)
