import math
import time
from typing import Any, Dict, List, Optional, Union

import lightning as L
import torch
from lightning.pytorch.utilities.data import extract_batch_size

# What every epoch of a trial reports to Ray Tune, beside its metrics
EPOCH_TRAIN_S = "epoch_train_s"
EPOCH_VALIDATION_S = "epoch_validation_s"
EPOCH_SAMPLES = "epoch_samples"
TRAIN_SAMPLES_PER_S = "train_samples_per_s"
SETUP_S = "setup_s"

# The key of the trial configuration that carries when the trial started
TRIAL_STARTED_AT = "trial_started_at"


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
