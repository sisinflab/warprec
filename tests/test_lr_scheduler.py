"""ReduceLROnPlateau, the one scheduler that steps on a monitored value.

Lightning refuses to step it unless the optimiser configuration names the value
to watch. It watches the mean training loss of the epoch: the one quantity
every training run has, in a sweep trial and in the retraining after
cross-validation alike, so the schedule a trial found is the schedule the
retrained model follows.
"""

import lightning as L
import pytest
from pydantic import ValidationError

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.config.model_configuration import LRSchedulerConfig

from conftest import make_model


def _fit(model: IterativeRecommender, dataset: Dataset, epochs: int) -> L.Trainer:
    """Train a model for some epochs, with no validation.

    Args:
        model (IterativeRecommender): The model to train.
        dataset (Dataset): The dataset to train on.
        epochs (int): The number of epochs.

    Returns:
        L.Trainer: The trainer after the fit.
    """
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(
        model,
        train_dataloaders=model.get_dataloader(
            interactions=dataset.train_set, sessions=dataset.train_session
        ),
    )
    return trainer


@pytest.mark.parametrize("model_name", ["BPR", "BSARec"])
def test_reduce_on_plateau_steps_on_the_training_loss(
    dataset: Dataset, model_name: str
):
    """The learning rate falls once the loss stops improving.

    An absolute threshold of 1e9 means no epoch after the first can count as
    an improvement, so with no patience every later epoch cuts the rate.
    BSARec does not log a loss of its own.
    """
    model = make_model(model_name, dataset)
    model.set_optimization_parameters(
        lr_scheduler_config=LRSchedulerConfig(
            name="ReduceLROnPlateau",
            params={
                "factor": 0.5,
                "patience": 0,
                "threshold": 1e9,
                "threshold_mode": "abs",
            },
        )
    )
    trainer = _fit(model, dataset, epochs=3)

    start = model.learning_rate
    final = trainer.optimizers[0].param_groups[0]["lr"]
    assert final == pytest.approx(start * 0.25)


def test_other_schedulers_are_unchanged(dataset: Dataset):
    """A scheduler that needs no monitored value steps as before."""
    model = make_model("BPR", dataset)
    model.set_optimization_parameters(
        lr_scheduler_config=LRSchedulerConfig(
            name="StepLR", params={"step_size": 1, "gamma": 0.5}
        )
    )
    trainer = _fit(model, dataset, epochs=2)

    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(
        model.learning_rate * 0.25
    )


def test_a_plateau_scheduler_cannot_maximise_the_loss():
    """The loss falls as the model improves, so 'max' would invert the schedule."""
    with pytest.raises(ValidationError, match="training loss"):
        LRSchedulerConfig(name="ReduceLROnPlateau", params={"mode": "max"})

    # Stating the default is fine
    LRSchedulerConfig(name="ReduceLROnPlateau", params={"mode": "min"})
