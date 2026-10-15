from typing import Optional

import ray
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.exceptions import GetTimeoutError, RayError

from warprec.recommenders.trainer.emissions import (
    TRACKER_TIMEOUT_SECONDS,
    EmissionsRecorder,
    EmissionsSettings,
)
from warprec.utils.logger import logger


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


class CodeCarbonCallback(tune.Callback):
    """CodeCarbon for a Ray Tune search.

    The run the search belongs to measures it: the train and swarm pipelines
    start an 'EmissionsRecorder' before they prepare the data and hand its
    'settings' over. A search run without one, from a 'Trainer' used on its
    own, starts its own recorder as its trials are set up and stops it when
    the search ends.

    With 'machine' tracking the recorder measures the Ray nodes, and there is
    nothing left for the search to do. With 'process' tracking each Ray Train
    worker of a trial measures itself (see 'trial_emissions') and hands its row
    to the recorder through Ray; this callback logs a trial that completes or
    fails with fewer rows delivered than it has workers.

    Args:
        save_to_api (bool): Whether to send the emissions to the CodeCarbon API.
        save_to_file (bool): Whether to write the rows to 'emissions.csv'.
        output_dir (str): The folder of 'emissions.csv'.
        tracking_mode (str): CodeCarbon's tracking mode, 'machine' or 'process'.
        project_name (Optional[str]): The project name of the rows, the run's
            name. Defaults to 'warprec'.
        settings (Optional[EmissionsSettings]): The measurement of the run the
            search belongs to, or None when the search measures itself.
    """

    def __init__(
        self,
        save_to_api: bool = False,
        save_to_file: bool = False,
        output_dir: str = "./",
        tracking_mode: str = "machine",
        project_name: Optional[str] = None,
        settings: Optional[EmissionsSettings] = None,
    ):
        self.save_to_api = save_to_api
        self.save_to_file = save_to_file
        self.output_dir = output_dir
        self.tracking_mode = tracking_mode
        self.project_name = project_name or "warprec"
        self.settings = settings
        self.workers_per_trial = 1
        self._recorder: Optional[EmissionsRecorder] = None

    def trial_settings(self, num_workers: int = 1) -> Optional[EmissionsSettings]:
        """What a trial needs to measure itself.

        Called as the search is set up: a search measuring itself starts
        doing so here.

        Args:
            num_workers (int): The Ray Train workers of each trial, as many
                rows as a trial delivers.

        Returns:
            Optional[EmissionsSettings]: The settings 'trial_emissions' takes,
                or None when the trials are not measured one by one.
        """
        self.workers_per_trial = num_workers
        if self.settings is None and self._recorder is None:
            self._recorder = EmissionsRecorder(
                tracking_mode=self.tracking_mode,
                save_to_api=self.save_to_api,
                save_to_file=self.save_to_file,
                output_dir=self.output_dir,
                project_name=self.project_name,
            )
            self._recorder.start()
        settings = self.settings
        if settings is None and self._recorder is not None:
            settings = self._recorder.settings
        if self.tracking_mode != "process":
            return None
        return settings

    def _check_rows(self, trial_id: str) -> None:
        """Log a finished trial whose workers did not all deliver their row.

        Args:
            trial_id (str): The trial.
        """
        settings = self.settings or (
            self._recorder.settings if self._recorder is not None else None
        )
        if self.tracking_mode != "process" or settings is None:
            return
        try:
            delivered = ray.get(
                settings.inbox.delivered.remote(trial_id),
                timeout=TRACKER_TIMEOUT_SECONDS,
            )
        except (RayError, GetTimeoutError) as error:
            logger.attention(
                f"Could not check the CodeCarbon rows of trial {trial_id}: "
                f"{type(error).__name__}."
            )
            return
        if delivered < self.workers_per_trial:
            logger.attention(
                f"Trial {trial_id} finished with {delivered} of its "
                f"{self.workers_per_trial} CodeCarbon rows delivered: the "
                "emissions of the missing workers are not in emissions.csv."
            )

    def on_trial_complete(self, iteration, trials, trial, **info):
        self._check_rows(trial.trial_id)

    def on_trial_error(self, iteration, trials, trial, **info):
        self._check_rows(trial.trial_id)

    def on_experiment_end(self, trials, **info):
        if self._recorder is not None:
            self._recorder.stop()
            self._recorder = None
