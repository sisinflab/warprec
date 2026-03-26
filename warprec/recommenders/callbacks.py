import os
import tempfile
from typing import Dict

import psutil
import torch
import lightning as L
from ray import train

from warprec.data.dataset import Dataset
from warprec.evaluation.evaluator import Evaluator


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
    """

    def __init__(self, evaluator: Evaluator, dataset: Dataset, strategy: str = "full"):
        super().__init__()
        self.evaluator = evaluator
        self.dataset = dataset
        self.strategy = strategy

    def on_train_epoch_end(self, trainer, pl_module):
        # Capture memory stats
        mem_stats = _get_memory_usage()
        for k, v in mem_stats.items():
            pl_module.log(k, v, on_epoch=True, prog_bar=False, sync_dist=True)

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
        for k, metrics in results.items():
            for metric_name, value in metrics.items():
                val_scalar = value.nanmean().item()
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

        # Create the checkpoint and report to Ray the results
        if trainer.is_global_zero:  # Execute only on master node
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")

                # Save the checkpoint using WarpRec custom information
                trainer.save_checkpoint(checkpoint_path, weights_only=True)

                # Ray reporting
                ray_checkpoint = train.Checkpoint.from_directory(temp_dir)
                train.report(metric_report, checkpoint=ray_checkpoint)
