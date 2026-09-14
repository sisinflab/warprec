# pylint: disable=unused-argument, protected-access
import os
from typing import Dict

from codecarbon import EmissionsTracker
from ray import tune
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


class CodeCarbonCallback(tune.Callback):
    """Custom CodeCarbon callback for Ray Tune.

    The tracker of a trial is stopped, and its emissions written, when the trial
    completes or fails, or when the experiment ends with the trial still
    running, as an interrupted sweep does.
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

    def on_trial_start(self, iteration, trials, trial, **info):
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

    def on_trial_error(self, iteration, trials, trial, **info):
        self._stop_tracker(trial.trial_id)

    def on_experiment_end(self, trials, **info):
        for trial_id in list(self.trackers):
            self._stop_tracker(trial_id)

    def _stop_tracker(self, trial_id):
        tracker = self.trackers.pop(trial_id, None)
        if tracker:
            tracker.stop()
