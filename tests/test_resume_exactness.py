"""A trial paused and resumed ends where the uninterrupted trial ends.

A trial checkpoint carries the weights, the optimizer and Lightning's loops,
but the random state training draws from used to be left behind: the loader's
shuffle order, the negatives its dataset samples (in this process or in
persistent workers), the global generators dropout and sampling layers use,
and the generators some models keep. A resumed trial drew its remaining epochs
from fresh generators, so a paused run ended on other weights, another best
iteration and other test results than the same run left alone.

The trials here run the train pipeline's objective in this process, with
Ray's reporting replaced by a callback that saves the checkpoint exactly where
RayTrainReportCallback does, and can stop the trial right after it.
"""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import lightning as L
import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.trainer import objectives
from warprec.recommenders.trainer import trainer as trainer_module
from warprec.utils.config.evaluation_configuration import EvaluationConfig
from warprec.utils.registry import params_registry

from conftest import build_params

EPOCHS = 4


class Paused(Exception):
    """Stands for the trial process being stopped by a pause."""


class SavingReport(L.Callback):
    """Saves the trial checkpoint where Ray's report callback saves it.

    Args:
        folder (Path): Where the checkpoint of every epoch goes.
        pause_after (Optional[int]): The epoch after whose checkpoint the trial
            stops, or None to let it finish.

    Attributes:
        CHECKPOINT_NAME (str): The file name the objective resumes from.
        final_weights (Dict[str, torch.Tensor]): The weights the last
            finished session ended on.
    """

    CHECKPOINT_NAME: str = "checkpoint.ckpt"
    final_weights: Dict[str, torch.Tensor] = {}

    def __init__(self, folder: Path, pause_after: Optional[int]):
        self.folder = folder
        self.pause_after = pause_after

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: Any) -> None:
        """Save this epoch's checkpoint, then stop if the pause is due.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (Any): The model.

        Raises:
            Paused: After the checkpoint of the epoch the pause falls on.
        """
        target = self.folder / str(trainer.current_epoch)
        target.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(target / self.CHECKPOINT_NAME, weights_only=False)
        if trainer.current_epoch == self.pause_after:
            raise Paused()

    def on_train_end(self, trainer: L.Trainer, pl_module: Any) -> None:
        """Keep the weights the trial ended on.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (Any): The model.
        """
        SavingReport.final_weights = {
            name: value.detach().clone()
            for name, value in pl_module.state_dict().items()
        }


def _trial_config(
    monkeypatch: pytest.MonkeyPatch,
    dataset: Dataset,
    tmp_path: Path,
    model_name: str,
    optimization: Dict[str, Any],
) -> dict:
    """The configuration the train pipeline hands to a trial of one model.

    Args:
        monkeypatch (pytest.MonkeyPatch): Keeps Ray out of the test.
        dataset (Dataset): The dataset to train on.
        tmp_path (Path): Where Ray Tune would store its results.
        model_name (str): The model of the trial.
        optimization (Dict[str, Any]): The model's optimization block.

    Returns:
        dict: The trial configuration, hyperparameters included.
    """
    captured: Dict[str, Any] = {}
    monkeypatch.setattr(trainer_module.ray, "put", lambda value: value)
    monkeypatch.setattr(
        trainer_module.tune,
        "with_parameters",
        lambda fn, **kwargs: captured.update(kwargs) or fn,
    )
    monkeypatch.setattr(trainer_module.tune, "with_resources", lambda fn, resources: fn)
    monkeypatch.setattr(
        trainer_module.Trainer, "_build_or_restore_tuner", lambda self, **kw: None
    )

    hyperparameters = {**build_params(model_name), "epochs": EPOCHS}
    params = params_registry.get(
        model_name,
        **hyperparameters,
        optimization={**optimization, "max_concurrent_trials": 1},
    )
    trainer_module.Trainer(storage_path=str(tmp_path))._setup_tuner(
        model_name=model_name,
        params=params,
        dataset=dataset,
        evaluation=EvaluationConfig(
            top_k=[5], metrics=["nDCG"], validation_metric="nDCG@5"
        ),
        validation_score="nDCG@5",
        device="cpu",
        ray_verbose=0,
    )
    return {**captured["data_bundle"], "params": hyperparameters}


def _run_trial(
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
    folder: Path,
    pause_after: Optional[int] = None,
    resume_from: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    """Run one session of a trial, the way a Ray Train worker runs it.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces Ray Train's session.
        config (dict): The trial configuration.
        folder (Path): Where the session saves its checkpoints.
        pause_after (Optional[int]): The epoch after which the session stops.
        resume_from (Optional[Path]): The checkpoint folder to resume from.

    Returns:
        Dict[str, torch.Tensor]: The weights the session ended on, empty when
            it was paused.
    """
    monkeypatch.setattr(objectives.ray, "get", lambda value: value)
    monkeypatch.setattr(
        objectives.train,
        "get_context",
        lambda: SimpleNamespace(get_world_size=lambda: 1),
    )
    checkpoint = (
        SimpleNamespace(as_directory=lambda: nullcontext(str(resume_from)))
        if resume_from is not None
        else None
    )
    monkeypatch.setattr(objectives.train, "get_checkpoint", lambda: checkpoint)
    monkeypatch.setattr(objectives.train, "report", lambda *a, **k: None)
    monkeypatch.setattr(
        objectives,
        "RayTrainReportCallback",
        type(
            "Report",
            (SavingReport,),
            {"__init__": lambda self: SavingReport.__init__(self, folder, pause_after)},
        ),
    )
    SavingReport.final_weights = {}
    objectives.objective_function(dict(config))
    return SavingReport.final_weights


def _assert_same_weights(left: Dict[str, torch.Tensor], right: Dict[str, torch.Tensor]):
    """Fail unless two sets of weights are bit-identical.

    Args:
        left (Dict[str, torch.Tensor]): The weights of one trial.
        right (Dict[str, torch.Tensor]): The weights of the other.
    """
    assert left and right, "a trial did not finish"
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), f"{name} differs"


@pytest.mark.parametrize(
    "model_name, optimization, pause_after",
    [
        # Shuffle order and negatives drawn in this process, paused after the
        # first epoch and after a later one
        ("BPR", {"num_workers": 0}, 0),
        ("BPR", {"num_workers": 0}, 2),
        # Negatives drawn in a persistent worker
        ("BPR", {"num_workers": 1}, 1),
        # The global generator: dropout and the reparameterisation noise
        ("MultiVAE", {"num_workers": 0}, 1),
        # A generator the model keeps for its knowledge-graph triples
        ("CKE", {"num_workers": 0}, 1),
        # A plateau scheduler, stepped after the epoch's checkpoint is written
        (
            "BPR",
            {
                "num_workers": 0,
                "lr_scheduler": {
                    "name": "ReduceLROnPlateau",
                    "params": {
                        "factor": 0.5,
                        "patience": 0,
                        "threshold": 1e9,
                        "threshold_mode": "abs",
                    },
                },
            },
            1,
        ),
    ],
)
def test_a_resumed_trial_ends_on_the_uninterrupted_weights(
    monkeypatch: pytest.MonkeyPatch,
    dataset: Dataset,
    tmp_path: Path,
    model_name: str,
    optimization: Dict[str, Any],
    pause_after: int,
):
    """Pausing after any epoch and resuming changes nothing the trial learns."""
    config = _trial_config(monkeypatch, dataset, tmp_path, model_name, optimization)

    uninterrupted = _run_trial(monkeypatch, config, tmp_path / "uninterrupted")

    paused = tmp_path / "paused"
    assert not _run_trial(monkeypatch, config, paused, pause_after=pause_after)
    resumed = _run_trial(
        monkeypatch, config, paused, resume_from=paused / str(pause_after)
    )

    _assert_same_weights(uninterrupted, resumed)
