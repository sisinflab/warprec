import os
from typing import Any, Dict, Optional

import psutil
import torch
import lightning as L
from ray import train
from torch import Tensor

from warprec.data.dataset import Dataset
from warprec.evaluation.evaluator import Evaluator
from warprec.utils.config.model_configuration import EarlyStopping
from warprec.utils.logger import logger

COMPLETED_EPOCHS = "completed_epochs"


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
            self.es_best_score = None
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

    def on_train_start(self, trainer, pl_module):  # pylint: disable=unused-argument
        """PyTorch Lightning hook used to report the epoch a trial resumes from.

        Ray Tune never receives the report of an epoch that ends while a run is
        pausing, although its checkpoint is saved. When a later session resumes
        from that checkpoint and the epoch was not logged, it is reported once.
        """
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

        # Early stopping logic
        if self.early_stopping_config and self.validation_score in metric_report:
            current_score = metric_report[self.validation_score]
            epoch = trainer.current_epoch

            if epoch >= self.grace_period:
                if self.es_best_score is None:
                    self.es_best_score = current_score
                else:
                    improved = (
                        (current_score < self.es_best_score - self.min_delta)
                        if self.mode == "min"
                        else (current_score > self.es_best_score + self.min_delta)
                    )

                    if improved:
                        self.es_best_score = current_score
                        self.wait = 0
                    else:
                        self.wait += 1

                    if self.wait >= self.patience:
                        if trainer.is_global_zero:
                            logger.attention(
                                f"Early stopping triggered at epoch {epoch} "
                                f"(Score: {current_score:.4f}, Best: {self.es_best_score:.4f})."
                            )
                        # This trigger will stop Lightning Trainer
                        trainer.should_stop = True

        # Log the remaining metrics to Lightning
        # NOTE: We don't need synching here as we already did it before, and
        # re-logging a key with a different sync_dist is rejected by Lightning
        for key, val in metric_report.items():
            if key in synced_keys:
                continue
            pl_module.log(key, val, prog_bar=True, on_epoch=True, sync_dist=False)
