import os
import random
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import psutil
import torch
import lightning as L
from ray import train
from torch import Tensor
from torch.utils.data import DataLoader

from warprec.data.dataset import Dataset
from warprec.data.entities.common import (
    get_generator_state,
    loader_random_state,
    prime_loader_workers,
    restore_loader_random_state,
    set_generator_state,
)
from warprec.evaluation.evaluator import Evaluator
from warprec.recommenders.base_recommender import TRAIN_LOSS
from warprec.utils.config.model_configuration import EarlyStopping
from warprec.utils.logger import logger

COMPLETED_EPOCHS = "completed_epochs"
RANDOM_STATE = "warprec_random_state"
_GENERATOR_TYPES = (torch.Generator, np.random.Generator, np.random.RandomState)


def _get_memory_usage() -> Dict[str, float]:
    """Calculates and returns a dictionary with peak RAM and VRAM usage.

    Returns:
        Dict[str, float]: The memory report.
    """
    report = {}
    process = psutil.Process(os.getpid())
    report["ram_peak_mb"] = process.memory_info().rss / 1024**2

    if torch.cuda.is_available():
        # This captures the peak for the current device assigned by Ray
        report["vram_peak_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    return report


class WarpRecLightningIntegrationCallback(L.Callback):
    # pylint: disable = too-many-instance-attributes  # this class is the state it holds
    """PyTorch Lightning callback implementation using WarpRec's custom Evaluator.

    Args:
        evaluator (Evaluator): WarpRec Evaluator instance.
        dataset (Dataset): The dataset used in the training process.
        strategy (str): The evaluation strategy to use.
        early_stopping_config (Optional[EarlyStopping]): The configuration of the early stopping.
        validation_score (str): The score used to validate the model.
        mode (str): The mode of optimization.
        last_logged_epoch (Optional[int]): The last epoch Ray Tune logged for the
            trial in a previous session, or None when it logged none.
        session_id (Optional[str]): The id of the session running the sweep.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        dataset: Dataset,
        strategy: str = "full",
        early_stopping_config: Optional[EarlyStopping] = None,
        validation_score: str = "nDCG@10",
        mode: str = "max",
        last_logged_epoch: Optional[int] = None,
        session_id: Optional[str] = None,
    ):
        super().__init__()
        self.evaluator = evaluator
        self.dataset = dataset
        self.strategy = strategy
        self.last_logged_epoch = last_logged_epoch
        self.session_id = session_id
        self.last_report: Optional[Dict[str, Any]] = None
        self.resumed_report: Optional[Dict[str, Any]] = None
        self.resumed_session: Optional[str] = None

        # Early Stopping configuration
        self.early_stopping_config = early_stopping_config
        self.validation_score = validation_score
        self.mode = mode

        # Best score tracking
        self.absolute_best_score = -float("inf") if self.mode == "max" else float("inf")

        if self.early_stopping_config:
            self.patience = self.early_stopping_config.patience
            self.min_delta = self.early_stopping_config.min_delta
            self.grace_period = self.early_stopping_config.grace_period
            self.es_best_score: Optional[float] = None
            self.wait = 0

    def state_dict(self) -> Dict[str, Any]:
        """PyTorch Lightning hook used to save the callback state.

        Returns:
            Dict[str, Any]: The best score, the early stopping state and the
                report of the epoch, with the session that wrote it.
        """
        state = {
            "absolute_best_score": self.absolute_best_score,
            "last_report": self.last_report,
            "session_id": self.session_id,
        }
        if self.early_stopping_config:
            state.update(es_best_score=self.es_best_score, wait=self.wait)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """PyTorch Lightning hook used to restore the callback state.

        Args:
            state_dict (Dict[str, Any]): The state saved by 'state_dict'.
        """
        self.absolute_best_score = state_dict["absolute_best_score"]
        self.resumed_report = state_dict.get("last_report")
        self.resumed_session = state_dict.get("session_id")
        if self.early_stopping_config:
            self.es_best_score = state_dict.get("es_best_score")
            self.wait = state_dict.get("wait", 0)

    def on_fit_start(self, trainer, pl_module):  # pylint: disable=unused-argument
        """PyTorch Lightning hook used to restore what a trial did before a pause.

        Ray Tune never receives the report of an epoch that ends while a run is
        pausing, although its checkpoint is saved. When a later session resumes
        from that checkpoint and the epoch was not logged, it is reported once.
        A stop already decided by early stopping is restored too, since Lightning
        does not save it. Both happen before the fit loop starts, which skips
        training entirely when a trial resumes from its last epoch.
        """
        if self.early_stopping_config and self.wait >= self.patience:
            trainer.should_stop = True
        report = self.resumed_report
        if (
            report is not None
            and self.resumed_session != self.session_id
            and (
                self.last_logged_epoch is None
                or report["epoch"] > self.last_logged_epoch
            )
        ):
            train.report(metrics=report)

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log(
            COMPLETED_EPOCHS, trainer.current_epoch + 1, on_epoch=True, sync_dist=True
        )

        # Capture memory stats
        mem_stats = _get_memory_usage()
        for k, v in mem_stats.items():
            pl_module.log(k, v, on_epoch=True, prog_bar=False, sync_dist=True)

        self.last_report = {k: v.item() for k, v in trainer.callback_metrics.items()}
        self.last_report.update(epoch=trainer.current_epoch, step=trainer.global_step)

        # Early stopping on the loss: every epoch has one, evaluated or not
        train_loss = trainer.callback_metrics.get(TRAIN_LOSS)
        if (
            self.early_stopping_config
            and self.early_stopping_config.monitor == "loss"
            and train_loss is not None
        ):
            self._early_stopping_step(
                trainer, float(train_loss), lower_is_better=True, label="Loss"
            )

        super().on_train_epoch_end(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loader = trainer.val_dataloaders

        # Safety check
        if val_loader is None:
            return

        # Evaluate the model and compute results
        self.evaluator.evaluate(
            model=pl_module,
            dataloader=val_loader,
            strategy=self.strategy,
            dataset=self.dataset,
            device=str(pl_module.device),
            verbose=False,
        )
        results = self.evaluator.compute_results()

        # Correctly format the Lightning logging
        metric_report = {}
        synced_keys = set()
        for k, metrics in results.items():
            for metric_name, value in metrics.items():
                val_scalar = (
                    value.nanmean().item() if isinstance(value, Tensor) else value
                )
                metric_key = f"{metric_name}@{k}"

                # Lightning logging
                pl_module.log(
                    metric_key,
                    val_scalar,
                    prog_bar=True,
                    on_epoch=True,
                    sync_dist=True,
                )

                # Store metric for Ray reporting
                metric_report[metric_key] = val_scalar
                synced_keys.add(metric_key)

        # Retrieve callback metrics from Lightning Trainer
        for key, val in trainer.callback_metrics.items():
            if isinstance(val, torch.Tensor):
                metric_report[key] = val.item()
            else:
                metric_report[key] = val

        # Keep track of the best validation score
        current_score = metric_report.get(self.validation_score)
        if current_score is not None:
            if self.mode == "max":
                self.absolute_best_score = max(self.absolute_best_score, current_score)
            else:
                self.absolute_best_score = min(self.absolute_best_score, current_score)

            metric_report[f"best_{self.validation_score}"] = self.absolute_best_score

        # Early stopping on the validation metric
        if (
            self.early_stopping_config
            and self.early_stopping_config.monitor != "loss"
            and self.validation_score in metric_report
        ):
            self._early_stopping_step(
                trainer,
                metric_report[self.validation_score],
                lower_is_better=self.mode == "min",
                label="Score",
            )

        # Log the remaining metrics to Lightning
        # NOTE: We don't need synching here as we already did it before, and
        # re-logging a key with a different sync_dist is rejected by Lightning
        for key, val in metric_report.items():
            if key in synced_keys:
                continue
            pl_module.log(key, val, prog_bar=True, on_epoch=True, sync_dist=False)

    def _early_stopping_step(
        self, trainer: L.Trainer, current: float, lower_is_better: bool, label: str
    ) -> None:
        """Count one more epoch towards early stopping, and stop when it is due.

        Args:
            trainer (L.Trainer): The trainer to stop.
            current (float): The value of the monitored quantity this epoch.
            lower_is_better (bool): Whether a decrease is an improvement.
            label (str): What the value is, for the log line.
        """
        epoch = trainer.current_epoch
        if epoch < self.grace_period:
            return

        if self.es_best_score is None:
            self.es_best_score = current
            return

        improved = (
            current < self.es_best_score - self.min_delta
            if lower_is_better
            else current > self.es_best_score + self.min_delta
        )
        if improved:
            self.es_best_score = current
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            if trainer.is_global_zero:
                logger.attention(
                    f"Early stopping triggered at epoch {epoch} "
                    f"({label}: {current:.4f}, Best: {self.es_best_score:.4f})."
                )
            # This trigger will stop Lightning Trainer
            trainer.should_stop = True


def _model_generators(pl_module: L.LightningModule) -> Iterator[Tuple[str, Any]]:
    """The random generators a model keeps as attributes, by attribute path.

    Some models sample during training from a generator of their own (the
    triples of CKE and KGAT, the edges of KGIN and FREEDOM) rather than from
    the global one.

    Args:
        pl_module (L.LightningModule): The model.

    Yields:
        Tuple[str, Any]: The attribute path and the generator.
    """
    for module_name, module in pl_module.named_modules():
        for attribute, value in vars(module).items():
            if isinstance(value, _GENERATOR_TYPES):
                yield f"{module_name}.{attribute}", value


class RandomStateCallback(L.Callback):
    """Makes a resumed trial draw exactly what the uninterrupted one would have.

    A trial checkpoint holds the weights, the optimizer and the loops, but not
    the random state training draws from: the order the loader shuffles in, the
    negatives the dataset samples (in this process or in persistent workers),
    the global generators dropout and augmentation use, and the generators some
    models keep. A trial resumed from it drew its remaining epochs from fresh
    generators, so a paused run ended on other weights and other results than
    one never paused.

    This callback saves all of them with every checkpoint and puts them back
    before the first resumed epoch. It also repeats the step a plateau
    learning-rate scheduler takes after the checkpoint of an epoch is written,
    which the resumed trial would otherwise skip.

    Args:
        train_dataloader (Optional[DataLoader]): The loader the model trains on,
            needed to hand their state to persistent workers before they start.
    """

    def __init__(self, train_dataloader: Optional[DataLoader] = None):
        super().__init__()
        self.train_dataloader = train_dataloader
        self._pending: Optional[Dict[str, Any]] = None

    def _loader(self, trainer: L.Trainer) -> Optional[DataLoader]:
        """The training loader, as Lightning iterates it when it has one.

        Args:
            trainer (L.Trainer): The trainer.

        Returns:
            Optional[DataLoader]: The loader, None when there is none.
        """
        loader = trainer.train_dataloader
        return loader if isinstance(loader, DataLoader) else self.train_dataloader

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Adds the random state of training to a checkpoint being written.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The model.
            checkpoint (Dict[str, Any]): The checkpoint being written.
        """
        loader = self._loader(trainer)
        plateau = {
            config.monitor: float(trainer.callback_metrics[config.monitor])
            for config in trainer.lr_scheduler_configs
            if config.reduce_on_plateau and config.monitor in trainer.callback_metrics
        }
        checkpoint[RANDOM_STATE] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "model": {
                name: get_generator_state(generator)
                for name, generator in _model_generators(pl_module)
            },
            "loader": loader_random_state(loader) if loader is not None else None,
            "epoch": trainer.current_epoch,
            "plateau": plateau,
        }

    def on_load_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Keeps the random state of a checkpoint a trial resumes from.

        Persistent workers start before the first resumed epoch, so they are
        given their state here; everything else is restored at train start.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The model.
            checkpoint (Dict[str, Any]): The checkpoint being restored.
        """
        state = checkpoint.get(RANDOM_STATE)
        if state is None or trainer.world_size > 1:
            if trainer.is_global_zero:
                logger.attention(
                    "The checkpoint this trial resumes from carries no random "
                    "state it can restore (it predates WarpRec saving one, or "
                    "the trial trains on several workers), so the resumed "
                    "epochs will not draw the samples an uninterrupted run would."
                )
            return

        self._pending = state
        loader = self.train_dataloader
        if (
            loader is not None
            and state["loader"] is not None
            and not prime_loader_workers(loader, state["loader"])
        ):
            logger.attention(
                "The sampling state of the persistent data loader workers could "
                "not be saved, so the resumed epochs will draw other negatives "
                "than an uninterrupted run would."
            )

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Restores the saved random state before the first resumed epoch.

        Lightning has already created the iterator of that epoch when this
        runs, so it is created again from the restored shuffling generator,
        as it is at the start of every other epoch.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The model.
        """
        state, self._pending = self._pending, None
        if state is None:
            return

        loader = self._loader(trainer)
        if loader is not None and state["loader"] is not None:
            restore_loader_random_state(loader, state["loader"])
            fetcher = getattr(trainer.fit_loop, "_data_fetcher", None)
            if fetcher is not None:
                iter(fetcher)

        for name, generator in _model_generators(pl_module):
            if name in state["model"]:
                set_generator_state(generator, state["model"][name])

        # Lightning steps a plateau scheduler after the epoch's checkpoint is
        # written, so the checkpoint holds it one step behind
        for config in trainer.lr_scheduler_configs:
            value = state["plateau"].get(config.monitor)
            if (
                config.reduce_on_plateau
                and config.interval == "epoch"
                and value is not None
                and (state["epoch"] + 1) % config.frequency == 0
            ):
                pl_module.lr_scheduler_step(config.scheduler, torch.tensor(value))

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
