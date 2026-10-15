import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.utilities.data import extract_batch_size
from ray import tune

# What every epoch of a trial reports to Ray Tune, beside its metrics
EPOCH_TRAIN_S = "epoch_train_s"
EPOCH_VALIDATION_S = "epoch_validation_s"
EPOCH_SAMPLES = "epoch_samples"
TRAIN_SAMPLES_PER_S = "train_samples_per_s"
SETUP_S = "setup_s"

# The key of the trial configuration that carries when the trial started
TRIAL_STARTED_AT = "trial_started_at"

# The columns of the time report built from the trials' timing
BEST_SETUP = "Best Trial Setup Time (s)"
BEST_TRAIN = "Best Trial Train-Only Time (s)"
BEST_VALIDATION = "Best Trial Validation Time (s)"
BEST_TRAINING = "Best Trial Training Time (s)"
BEST_EPOCHS = "Best Trial Epochs"
BEST_MEAN_EPOCH_TRAIN = "Best Trial Mean Epoch Train Time (s)"
BEST_MIN_EPOCH_TRAIN = "Best Trial Min Epoch Train Time (s)"
BEST_MAX_EPOCH_TRAIN = "Best Trial Max Epoch Train Time (s)"
BEST_MEAN_EPOCH_VALIDATION = "Best Trial Mean Epoch Validation Time (s)"
BEST_THROUGHPUT = "Best Trial Train Throughput (samples/s)"
MEAN_SETUP = "Mean Trial Setup Time (s)"
MEAN_TRAIN = "Mean Trial Train-Only Time (s)"
MEAN_VALIDATION = "Mean Trial Validation Time (s)"
SEARCH_OVERHEAD = "Ray Tune Overhead (s)"
RETRAINING_TRAIN = "Retraining Train-Only Time (s)"


def synchronize(device: Union[str, torch.device, None]) -> None:
    """Waits for the work queued on a GPU, so that a timer read now is exact.

    CUDA runs kernels asynchronously: without this a timer stops when the
    work was queued, not when it was done.

    Args:
        device (Union[str, torch.device, None]): The device the work runs on.
    """
    if device is None or not torch.cuda.is_available():
        return
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def epoch_timing(
    train_s: float, validation_s: float, samples: float, setup_s: float
) -> Dict[str, float]:
    """The timing one epoch reports to Ray Tune.

    Args:
        train_s (float): The time spent training, validation excluded.
        validation_s (float): The time spent in the end-of-epoch evaluation.
        samples (float): The training examples the epoch went through.
        setup_s (float): The setup of the trial, the same for every epoch.

    Returns:
        Dict[str, float]: The epoch's timing, by result key.
    """
    return {
        EPOCH_TRAIN_S: train_s,
        EPOCH_VALIDATION_S: validation_s,
        EPOCH_SAMPLES: samples,
        TRAIN_SAMPLES_PER_S: samples / train_s if train_s > 0 else math.nan,
        SETUP_S: setup_s,
    }


def _batch_size(batch: Any) -> int:
    """The number of training examples in a batch.

    An example is whatever the model's loader yields one of: an interaction,
    a triplet, a sequence, a user row. Lightning reads it from the first
    dimension of the batch's first tensor.

    Args:
        batch (Any): The batch.

    Returns:
        int: The examples in it, 0 when it holds no tensor.
    """
    try:
        return int(extract_batch_size(batch))
    except Exception:  # pylint: disable=broad-exception-caught
        return 0


class EpochTimingCallback(L.Callback):
    # pylint: disable = too-many-instance-attributes  # this class is the state it holds
    """Times every training epoch apart from the validation that ends it.

    Lightning validates inside the training epoch, after its last batch, so
    the epoch's training time is its whole duration minus the validation in
    it. Each epoch logs the keys of 'epoch_timing', which reach Ray Tune with
    the epoch's report when this callback runs before the report callback.

    The setup is the time from the start of the trial to its first training
    batch: starting the Ray Train workers, fetching the dataset, building the
    model and its loaders, and Lightning's own preparation. The first
    epoch's training time starts with that batch.

    Under several workers every worker measures itself and Ray Tune keeps the
    report of the first; the samples are summed over the workers, so the
    throughput is that of the whole worker group.

    Args:
        started_at (Optional[float]): When the trial started, as 'time.time()'.
            When None, the setup is measured from the creation of the callback.
    """

    def __init__(self, started_at: Optional[float] = None):
        super().__init__()
        self.started_at = time.time() if started_at is None else started_at
        self.setup_s: Optional[float] = None
        self.epochs: List[Dict[str, float]] = []
        self._epoch_start = 0.0
        self._validation_s = 0.0
        self._validation_start: Optional[float] = None
        self._samples = 0

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Starts the clock of an epoch.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The model.
        """
        synchronize(pl_module.device)
        self._epoch_start = time.perf_counter()
        self._validation_s = 0.0
        self._validation_start = None
        self._samples = 0

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Counts the batch's examples, and ends the setup on the first batch.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The model.
            batch (Any): The batch.
            batch_idx (int): The index of the batch in the epoch.
        """
        if self.setup_s is None:
            synchronize(pl_module.device)
            self.setup_s = max(time.time() - self.started_at, 0.0)
            self._epoch_start = time.perf_counter()
        self._samples += _batch_size(batch)

    def on_validation_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Starts the clock of the end-of-epoch validation.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The model.
        """
        if trainer.sanity_checking:
            return
        synchronize(pl_module.device)
        self._validation_start = time.perf_counter()

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Stops the clock of the end-of-epoch validation.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The model.
        """
        if self._validation_start is None:
            return
        synchronize(pl_module.device)
        self._validation_s += time.perf_counter() - self._validation_start
        self._validation_start = None

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Logs the timing of the epoch that just ended.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The model.
        """
        synchronize(pl_module.device)
        elapsed = time.perf_counter() - self._epoch_start
        samples = trainer.strategy.reduce(
            torch.tensor(float(self._samples), device=pl_module.device),
            reduce_op="sum",
        )
        timing = epoch_timing(
            train_s=max(elapsed - self._validation_s, 0.0),
            validation_s=self._validation_s,
            samples=float(samples),
            setup_s=self.setup_s if self.setup_s is not None else math.nan,
        )
        self.epochs.append(timing)

        # Logged in double precision, which Lightning keeps: a count of
        # samples past 2**24 would lose its units in single precision
        for key, value in timing.items():
            pl_module.log(
                key,
                torch.tensor(value, dtype=torch.float64),
                on_epoch=True,
                batch_size=1,
                sync_dist=False,
            )


@dataclass
class TrialTiming:
    """Where the time of one trial went, epoch by epoch.

    Attributes:
        setup_s (float): The trial's setup.
        epoch_train_s (List[float]): The training time of each epoch.
        epoch_validation_s (List[float]): The validation time of each epoch,
            0 for an epoch that was not validated.
        samples (float): The training examples of all its epochs.
    """

    setup_s: float
    epoch_train_s: List[float] = field(default_factory=list)
    epoch_validation_s: List[float] = field(default_factory=list)
    samples: float = 0.0

    @property
    def train_s(self) -> float:
        """The time spent only training.

        Returns:
            float: The sum of the epochs' training time.
        """
        return float(sum(self.epoch_train_s))

    @property
    def validation_s(self) -> float:
        """The time spent in the end-of-epoch validation.

        Returns:
            float: The sum of the epochs' validation time.
        """
        return float(sum(self.epoch_validation_s))

    @property
    def busy_s(self) -> float:
        """The trial's measured work: setup, training and validation.

        Returns:
            float: Their sum.
        """
        return self.setup_s + self.train_s + self.validation_s


def trial_timing(history: Optional[pd.DataFrame]) -> Optional[TrialTiming]:
    """Reads the timing of a trial from the results it reported.

    An epoch reported twice, which a trial resumed after a pause can do, is
    counted once, with its last report.

    Args:
        history (Optional[pd.DataFrame]): The trial's results, one row per
            report, or None when it reported nothing.

    Returns:
        Optional[TrialTiming]: The timing, or None when the trial reported
            none (it failed before training).
    """
    if history is None or EPOCH_TRAIN_S not in history:
        return None
    rows = history[history[EPOCH_TRAIN_S].notna()]
    if rows.empty:
        return None
    if "epoch" in rows and rows["epoch"].notna().all():
        rows = rows.drop_duplicates(subset="epoch", keep="last")

    def column(key: str) -> List[float]:
        if key not in rows:
            return [0.0] * len(rows)
        return [float(v) for v in rows[key].fillna(0.0)]

    return TrialTiming(
        setup_s=float(rows[SETUP_S].iloc[-1]) if SETUP_S in rows else math.nan,
        epoch_train_s=column(EPOCH_TRAIN_S),
        epoch_validation_s=column(EPOCH_VALIDATION_S),
        samples=float(sum(column(EPOCH_SAMPLES))),
    )


def _mean(values: Iterable[float]) -> float:
    """The mean of some values.

    Args:
        values (Iterable[float]): The values.

    Returns:
        float: Their mean, NaN when there are none.
    """
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def summarise_timing(
    trials: Mapping[str, Optional[TrialTiming]], best: Iterable[str]
) -> Dict[str, float]:
    """The time report's columns, from the timing of every trial of a search.

    The best trial's columns are averaged over the trials given as best,
    which under cross-validation are the folds of the best configuration; its
    epoch statistics pool their epochs.

    Args:
        trials (Mapping[str, Optional[TrialTiming]]): The timing of each
            trial, None for a trial that reported none.
        best (Iterable[str]): The ids of the best trials.

    Returns:
        Dict[str, float]: The columns, NaN where no trial gives a value.
    """
    timed = [timing for timing in trials.values() if timing is not None]
    best_timed = [trials[i] for i in best if trials.get(i) is not None]
    best_timed = [timing for timing in best_timed if timing is not None]

    epochs_train = [t for timing in best_timed for t in timing.epoch_train_s]
    epochs_validated = [
        v for timing in best_timed for v in timing.epoch_validation_s if v > 0
    ]
    best_train = sum(timing.train_s for timing in best_timed)
    best_samples = sum(timing.samples for timing in best_timed)

    return {
        BEST_SETUP: _mean(timing.setup_s for timing in best_timed),
        BEST_TRAIN: _mean(timing.train_s for timing in best_timed),
        BEST_VALIDATION: _mean(timing.validation_s for timing in best_timed),
        BEST_TRAINING: _mean(
            timing.train_s + timing.validation_s for timing in best_timed
        ),
        BEST_EPOCHS: _mean(len(timing.epoch_train_s) for timing in best_timed),
        BEST_MEAN_EPOCH_TRAIN: _mean(epochs_train),
        BEST_MIN_EPOCH_TRAIN: min(epochs_train) if epochs_train else math.nan,
        BEST_MAX_EPOCH_TRAIN: max(epochs_train) if epochs_train else math.nan,
        BEST_MEAN_EPOCH_VALIDATION: _mean(epochs_validated),
        BEST_THROUGHPUT: best_samples / best_train if best_train > 0 else math.nan,
        MEAN_SETUP: _mean(timing.setup_s for timing in timed),
        MEAN_TRAIN: _mean(timing.train_s for timing in timed),
        MEAN_VALIDATION: _mean(timing.validation_s for timing in timed),
    }


class TrialSpans(tune.Callback):
    """Records, on the driver, when each trial of a search runs.

    It tells whether the trials ran one at a time, which is when the search's
    wall clock can be split honestly into the trials' work and the rest.
    'spans' holds the (start, end) of every run of each trial, by trial id,
    from 'time.perf_counter'; an end is NaN while the trial runs.
    """

    def __init__(self) -> None:
        self.spans: Dict[str, List[Tuple[float, float]]] = {}

    def on_trial_start(
        self, iteration: int, trials: List, trial: Any, **info: Any
    ) -> None:
        """Ray Tune hook: a trial starts, or starts again.

        Args:
            iteration (int): The Tune loop iteration.
            trials (List): Every trial of the search.
            trial (Any): The trial that starts.
            **info (Any): Further keyword arguments from Ray Tune.
        """
        self.spans.setdefault(trial.trial_id, []).append(
            (time.perf_counter(), math.nan)
        )

    def _end(self, trial: Any) -> None:
        """Closes the run of a trial that has just stopped.

        Args:
            trial (Any): The trial.
        """
        runs = self.spans.get(trial.trial_id)
        if runs and math.isnan(runs[-1][1]):
            runs[-1] = (runs[-1][0], time.perf_counter())

    def on_trial_complete(
        self, iteration: int, trials: List, trial: Any, **info: Any
    ) -> None:
        """Ray Tune hook: a trial finished.

        Args:
            iteration (int): The Tune loop iteration.
            trials (List): Every trial of the search.
            trial (Any): The trial that finished.
            **info (Any): Further keyword arguments from Ray Tune.
        """
        self._end(trial)

    def on_trial_error(
        self, iteration: int, trials: List, trial: Any, **info: Any
    ) -> None:
        """Ray Tune hook: a trial failed.

        Args:
            iteration (int): The Tune loop iteration.
            trials (List): Every trial of the search.
            trial (Any): The trial that failed.
            **info (Any): Further keyword arguments from Ray Tune.
        """
        self._end(trial)

    def ran_one_at_a_time(self) -> bool:
        """Whether every trial ran once, and none while another ran.

        Returns:
            bool: True when the trials ran strictly one after the other.
        """
        runs = []
        for trial_runs in self.spans.values():
            if len(trial_runs) != 1:
                return False
            runs.append(trial_runs[0])
        if any(math.isnan(end) for _, end in runs):
            return False
        runs.sort()
        return all(
            following[0] >= current[1] for current, following in zip(runs, runs[1:])
        )


def search_overhead(
    search_s: float,
    spans: TrialSpans,
    trials: Mapping[str, Optional[TrialTiming]],
) -> float:
    """The part of a search's wall clock no trial spent setting up or training.

    It is Ray Tune's own time: starting the search, scheduling the trials,
    saving checkpoints and passing results on. It can be told apart only
    when the trials ran one at a time and every one of them was timed.

    Args:
        search_s (float): The wall clock of the search.
        spans (TrialSpans): When each trial ran, seen from the driver.
        trials (Mapping[str, Optional[TrialTiming]]): The timing of each trial.

    Returns:
        float: The overhead in seconds, NaN when it cannot be computed.
    """
    if (
        not trials
        or set(spans.spans) != set(trials)
        or any(timing is None for timing in trials.values())
        or not spans.ran_one_at_a_time()
    ):
        return math.nan
    return search_s - sum(
        timing.busy_s for timing in trials.values() if timing is not None
    )
