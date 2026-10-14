# pylint: disable=unused-argument, protected-access
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Dict, Iterator, Optional

from codecarbon import EmissionsTracker
from codecarbon.output_methods.base_output import BaseOutput
from codecarbon.output_methods.emissions_data import EmissionsData
from codecarbon.output_methods.file import FileOutput
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback


class WarpRecWandbLoggerCallback(WandbLoggerCallback):
    """Ray's W&B callback, closing the runs of the trials still open at the end.

    Ray Tune stops the trials still running at the end of an interrupted sweep
    without calling 'on_trial_complete', so Ray's callback never tells their
    logging actors to finish, and 'on_experiment_end' waits 'upload_timeout',
    30 minutes by default, for them. Telling them first lets each one flush its
    queue and close its run, and the wait ends within seconds.
    """

    def on_experiment_end(self, trials, **info):
        if hasattr(self, "_signal_logging_actor_stop"):
            for trial in list(self._trial_logging_actors):
                self._signal_logging_actor_stop(trial=trial)
        super().on_experiment_end(trials, **info)


def mlflow_metric_name(name: str) -> str:
    """The name MLflow receives for a WarpRec metric.

    WarpRec names a metric ``<metric>@<k>``, and MLflow refuses names holding an
    ``@``, so the cutoff is separated by a slash instead (``nDCG@10`` is sent as
    ``nDCG/10``), which MLflow also uses to group related metrics.

    Args:
        name (str): The metric name as WarpRec reports it.

    Returns:
        str: The name to log the metric under in MLflow.
    """
    return name.replace("@", "/")


class WarpRecMLflowLoggerCallback(MLflowLoggerCallback):
    """Ray's MLflow callback, with metric names MLflow accepts.

    Only what is sent to MLflow is renamed: the report Ray Tune and the other
    callbacks see keeps WarpRec's ``<metric>@<k>`` names.
    """

    def log_trial_result(self, iteration, trial, result):
        renamed = {mlflow_metric_name(key): value for key, value in result.items()}
        super().log_trial_result(iteration, trial, renamed)


TRIAL_EMISSIONS_DIR = ".codecarbon_trials"


class _TrialRow(BaseOutput):
    """Keep the row a trial's tracker measured, for the driver to write.

    CodeCarbon samples the CPU and RAM utilization and the RAM used every
    second, and averages them into the row, only while its tracker has an
    output to write to. The trial's tracker writes nowhere itself, so without
    one the row read 0 for all three.

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


@contextmanager
def trial_emissions(
    settings: Optional[Dict[str, Any]], trial_id: Optional[str]
) -> Iterator[None]:
    """Measure the emissions of the process a trial trains in.

    CodeCarbon's 'process' mode measures the process that starts the tracker,
    and the driver's Ray Tune callback runs in the driver, not where the trial
    trains. With 'process' tracking the trial therefore measures itself. It
    sends its emissions to the CodeCarbon API itself, when asked to, and leaves
    them for the driver's callback to append to 'emissions.csv', so that one
    process writes the file.

    Args:
        settings (Optional[Dict[str, Any]]): What 'CodeCarbonCallback.trial_settings'
            returned, or None when the trial is not measured.
        trial_id (Optional[str]): The Ray Tune trial, the project name of the row.

    Yields:
        None: While the trial trains.
    """
    if settings is None:
        yield
        return

    row = _TrialRow()
    tracker = EmissionsTracker(
        project_name=trial_id,
        save_to_api=settings["save_to_api"],
        save_to_file=False,
        output_dir=settings["output_dir"],
        tracking_mode="process",
        log_level="error",
        output_handlers=[row],
    )
    tracker.start()
    try:
        yield
    finally:
        tracker.stop()
        data = row.total
        if settings["save_to_file"] and data is not None:
            folder = os.path.join(settings["output_dir"], TRIAL_EMISSIONS_DIR)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{trial_id}-{uuid.uuid4().hex}.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump(asdict(data), file)


class CodeCarbonCallback(tune.Callback):
    """Custom CodeCarbon callback for Ray Tune.

    The tracker of a trial is stopped, and its emissions written, when the trial
    completes or fails, or when the experiment ends with the trial still
    running, as an interrupted sweep does.

    With 'machine' tracking the trackers run here, on the driver, since the
    machine is the same from any process on it. With 'process' tracking each
    trial measures the process it trains in (see 'trial_emissions'), and this
    callback only appends the rows the trials leave to 'emissions.csv'.
    """

    def __init__(
        self,
        save_to_api=False,
        save_to_file=False,
        output_dir="./",
        tracking_mode="machine",
    ):
        self.save_to_api = save_to_api
        self.save_to_file = save_to_file
        self.output_dir = output_dir
        self.tracking_mode = tracking_mode
        self.trackers: Dict[str, EmissionsTracker] = {}
        os.makedirs(self.output_dir, exist_ok=True)

    def trial_settings(self) -> Optional[Dict[str, Any]]:
        """What a trial needs to measure itself.

        Returns:
            Optional[Dict[str, Any]]: The settings 'trial_emissions' takes, or
                None when the trials are measured from the driver.
        """
        if self.tracking_mode != "process":
            return None
        return {
            "save_to_api": self.save_to_api,
            "save_to_file": self.save_to_file,
            "output_dir": self.output_dir,
        }

    def _write_trial_rows(self, trial_id: Optional[str] = None) -> None:
        """Append to 'emissions.csv' the rows trials measured themselves.

        Args:
            trial_id (Optional[str]): The trial whose rows to write, or None
                for every trial.
        """
        folder = os.path.join(self.output_dir, TRIAL_EMISSIONS_DIR)
        if not os.path.isdir(folder):
            return
        output = FileOutput("emissions.csv", self.output_dir, on_csv_write="append")
        for name in sorted(os.listdir(folder)):
            if trial_id is not None and not name.startswith(f"{trial_id}-"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as file:
                data = EmissionsData(**json.load(file))
            output.out(data, data)
            os.remove(path)
        if not os.listdir(folder):
            os.rmdir(folder)

    def on_trial_start(self, iteration, trials, trial, **info):
        if self.tracking_mode == "process":
            return
        tracker = EmissionsTracker(
            project_name=trial.trial_id,  # Tag each row with its trial
            save_to_api=self.save_to_api,
            save_to_file=self.save_to_file,
            output_dir=self.output_dir,
            tracking_mode=self.tracking_mode,
            log_level="error",  # Reduce noise
        )
        tracker.start()
        self.trackers[trial.trial_id] = tracker

    def on_trial_complete(self, iteration, trials, trial, **info):
        self._stop_tracker(trial.trial_id)
        if self.save_to_file:
            self._write_trial_rows(trial.trial_id)

    def on_trial_error(self, iteration, trials, trial, **info):
        self._stop_tracker(trial.trial_id)
        if self.save_to_file:
            self._write_trial_rows(trial.trial_id)

    def on_experiment_end(self, trials, **info):
        for trial_id in list(self.trackers):
            self._stop_tracker(trial_id)
        if self.save_to_file:
            self._write_trial_rows()

    def _stop_tracker(self, trial_id):
        tracker = self.trackers.pop(trial_id, None)
        if tracker:
            tracker.stop()
