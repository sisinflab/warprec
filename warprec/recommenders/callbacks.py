import lightning as L

from warprec.data.dataset import Dataset
from warprec.evaluation.evaluator import Evaluator


class EvaluationCallback(L.Callback):
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
        for k, metrics in results.items():
            for metric_name, value in metrics.items():
                pl_module.log(
                    f"{metric_name}@{k}",
                    value.nanmean().item(),
                    prog_bar=True,
                    on_epoch=True,
                )
