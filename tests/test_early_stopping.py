"""What early stopping watches, and that it stops a trial when it should.

The configuration offers two things to monitor: the validation metric
('score') and the training loss ('loss'). Both have to steer the trial; a
setting that is validated and then ignored is worse than no setting at all.
"""

from types import SimpleNamespace
from typing import List

import lightning as L
import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import TRAIN_LOSS
from warprec.recommenders.callbacks import WarpRecLightningIntegrationCallback
from warprec.utils.config.model_configuration import EarlyStopping

from conftest import make_model


def _callback(
    monitor: str, patience: int, mode: str = "max", min_delta: float = 0.0
) -> WarpRecLightningIntegrationCallback:
    """An integration callback that only early-stops; it never evaluates.

    Args:
        monitor (str): What early stopping watches.
        patience (int): Epochs without improvement before stopping.
        mode (str): The direction of the validation metric.
        min_delta (float): The improvement that counts.

    Returns:
        WarpRecLightningIntegrationCallback: The callback.
    """
    return WarpRecLightningIntegrationCallback(
        evaluator=None,  # type: ignore[arg-type]
        dataset=None,  # type: ignore[arg-type]
        early_stopping_config=EarlyStopping(
            monitor=monitor, patience=patience, min_delta=min_delta
        ),
        validation_score="nDCG@10",
        mode=mode,
    )


def _feed_losses(
    callback: WarpRecLightningIntegrationCallback, losses: List[float]
) -> List[bool]:
    """Run the callback's epoch end over a sequence of epoch losses.

    Args:
        callback (WarpRecLightningIntegrationCallback): The callback under test.
        losses (List[float]): The mean training loss of each epoch.

    Returns:
        List[bool]: Whether the trainer was told to stop after each epoch.
    """
    module = SimpleNamespace(log=lambda *args, **kwargs: None)
    stops = []
    for epoch, loss in enumerate(losses):
        trainer = SimpleNamespace(
            current_epoch=epoch,
            global_step=epoch,
            callback_metrics={TRAIN_LOSS: torch.tensor(loss)},
            should_stop=False,
            is_global_zero=True,
        )
        callback.on_train_epoch_end(trainer, module)
        stops.append(trainer.should_stop)
    return stops


@pytest.mark.parametrize("mode", ["max", "min"])
def test_a_falling_loss_never_stops_the_trial(mode: str):
    """Lower loss is better whatever direction the validation metric has."""
    stops = _feed_losses(_callback("loss", patience=1, mode=mode), [5, 4, 3, 2, 1])
    assert not any(stops)


@pytest.mark.parametrize("mode", ["max", "min"])
def test_a_loss_that_stops_falling_stops_the_trial(mode: str):
    """Patience counts the epochs whose loss did not improve on the best."""
    stops = _feed_losses(_callback("loss", patience=2, mode=mode), [5.0, 4.0, 4.5, 4.2])
    assert stops == [False, False, False, True]


def test_score_monitoring_ignores_the_loss():
    """The default keeps watching the validation metric only."""
    stops = _feed_losses(_callback("score", patience=1), [1.0, 2.0, 3.0, 4.0])
    assert not any(stops)


@pytest.mark.parametrize("model_name", ["BPR", "BSARec"])
def test_loss_monitoring_stops_a_real_fit(dataset: Dataset, model_name: str):
    """Early stopping on the loss acts in a real Lightning fit.

    No validation runs at all, so only the loss can stop the trial. A huge
    min_delta makes every epoch after the first a non-improvement. BSARec does
    not log its own loss, so it shows the loss is collected for every model.
    """
    model = make_model(model_name, dataset)
    callback = _callback("loss", patience=2, min_delta=1e9)
    trainer = L.Trainer(
        max_epochs=20,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    trainer.fit(
        model,
        train_dataloaders=model.get_dataloader(
            interactions=dataset.train_set, sessions=dataset.train_session
        ),
    )

    assert TRAIN_LOSS in trainer.callback_metrics
    assert torch.isfinite(trainer.callback_metrics[TRAIN_LOSS])
    # Epoch 0 sets the best, epochs 1 and 2 do not improve on it
    assert trainer.current_epoch == 3
