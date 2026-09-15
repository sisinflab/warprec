import csv
import json
from pathlib import Path
from typing import Dict

import pyarrow.fs
from ray.air.constants import EXPR_PROGRESS_FILE, EXPR_RESULT_FILE
from ray.train._internal.storage import get_fs_and_path
from ray.tune.logger import CSVLoggerCallback, JsonLoggerCallback


def read_logged_epochs(experiment_path: str) -> Dict[str, int]:
    """Reads the last epoch Ray Tune logged for each trial of a stored experiment.

    Args:
        experiment_path (str): The path of the Ray Tune experiment in storage.

    Returns:
        Dict[str, int]: The last logged epoch of each trial, by trial id. Trials
            that logged no epoch are left out.
    """
    epochs: Dict[str, int] = {}
    filesystem, path = get_fs_and_path(experiment_path)
    selector = pyarrow.fs.FileSelector(path, allow_not_found=True)
    for trial_dir in filesystem.get_file_info(selector):
        result_file = f"{trial_dir.path}/{EXPR_RESULT_FILE}"
        if filesystem.get_file_info(result_file).type != pyarrow.fs.FileType.File:
            continue
        with filesystem.open_input_stream(result_file) as stream:
            lines = stream.read().decode(errors="replace").splitlines()
        for line in reversed(lines):
            try:
                report = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in report:
                epochs[report["trial_id"]] = int(report["epoch"])
                break
    return epochs


class _StoredHistoryRestore:  # pylint: disable=too-few-public-methods
    """Restores the results a trial wrote before a pause when it is resumed.

    Ray Tune copies a trial's result file back from storage only when the trial
    has a Tune checkpoint. A WarpRec trial never has one, since its checkpoints
    belong to the Ray Train run inside it, so a resumed trial would start an
    empty file and the next sync would overwrite its history in storage.

    A run killed outright can leave the last line of a stored file half written.
    That line is dropped, so that new results do not continue it.
    """

    def _restore_from_remote(self, file_name, trial):
        local_file = Path(trial.local_path, file_name)
        if local_file.exists():
            return
        stored_file = Path(trial.storage.trial_fs_path, file_name).as_posix()
        filesystem = trial.storage.storage_filesystem
        if filesystem.get_file_info(stored_file).type != pyarrow.fs.FileType.File:
            return
        pyarrow.fs.copy_files(
            stored_file, local_file.as_posix(), source_filesystem=filesystem
        )
        content = local_file.read_bytes()
        if not content.endswith(b"\n"):
            local_file.write_bytes(content[: content.rfind(b"\n") + 1])


class WarpRecJsonLoggerCallback(_StoredHistoryRestore, JsonLoggerCallback):
    """Ray Tune's 'result.json' logger, keeping the history of resumed trials."""


class WarpRecCSVLoggerCallback(_StoredHistoryRestore, CSVLoggerCallback):
    """Ray Tune's 'progress.csv' logger, keeping the history of resumed trials.

    A resumed file keeps the columns of its stored header, so that a report
    with different keys cannot shift the values under the wrong columns.
    """

    def _setup_trial(self, trial):
        super()._setup_trial(trial)
        if self._trial_continue[trial]:
            progress_file = Path(trial.local_path, EXPR_PROGRESS_FILE)
            with progress_file.open(encoding="utf-8", newline="") as stored:
                header = next(csv.reader(stored))
            self._trial_csv[trial] = csv.DictWriter(self._trial_files[trial], header)
