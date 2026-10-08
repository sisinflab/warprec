"""Behavioural tests for what a resumed run reuses and what it starts afresh.

The manifest decides whether a model's progress may be continued. Ray Tune
keeps its own copy of that progress on disk, so a decision to discard the
manifest's record is only honoured if the Ray Tune experiment is not quietly
restored behind its back.
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from warprec.common.run_state import (
    ModelState,
    ModelStatus,
    RunState,
    model_fingerprint,
    reconcile_model_state,
)
from warprec.recommenders.trainer import trainer as trainer_module
from warprec.recommenders.trainer.trainer import Trainer

PARAMS: Dict[str, Any] = {"l2": [10, 100], "optimization": {"strategy": "grid"}}
CHANGED: Dict[str, Any] = {"l2": [10, 500], "optimization": {"strategy": "grid"}}


def a_run(**models: ModelState) -> RunState:
    """A manifest holding the given model states.

    Args:
        **models (ModelState): The model states, keyed by model name.

    Returns:
        RunState: The manifest under test.
    """
    state = RunState(
        run_name="run",
        pipeline="train",
        warprec_version="2.1.2",
        writer_timestamp="20261008",
        config_fingerprint="abc",
    )
    state.models.update(models)
    return state


def a_ray_experiment(storage: Path, name: str) -> Path:
    """Lays down a directory Ray Tune would consider restorable.

    Args:
        storage (Path): The Ray storage path.
        name (str): The experiment name.

    Returns:
        Path: The experiment directory.
    """
    experiment = storage / name
    experiment.mkdir(parents=True)
    (experiment / "tuner.pkl").write_bytes(b"old experiment")
    return experiment


class FakeTuner:
    """Stands in for Ray Tune's Tuner and records what was asked of it."""

    built: List[Dict[str, Any]] = []
    restored: List[str] = []

    def __init__(self, *args: Any, **kwargs: Any):
        FakeTuner.built.append(kwargs)

    @staticmethod
    def can_restore(path: str) -> bool:
        """Mirrors Ray Tune: an experiment is restorable when its state file exists.

        Args:
            path (str): The experiment directory.

        Returns:
            bool: Whether the experiment can be restored.
        """
        return (Path(path) / "tuner.pkl").exists()

    @classmethod
    def restore(cls, path: str, **kwargs: Any) -> str:
        """Records a restore instead of performing it.

        Args:
            path (str): The experiment directory.
            **kwargs (Any): Ignored restore options.

        Returns:
            str: A marker for the restored tuner.
        """
        cls.restored.append(path)
        return "restored"


@pytest.fixture()
def fake_tuner(monkeypatch: pytest.MonkeyPatch) -> type:
    """Replaces Ray Tune's Tuner inside the trainer.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture.

    Returns:
        type: The fake tuner class.
    """
    FakeTuner.built = []
    FakeTuner.restored = []
    monkeypatch.setattr(trainer_module, "Tuner", FakeTuner)
    return FakeTuner


def build_tuner(trainer: Trainer, model_name: str) -> Any:
    """Runs the trainer's build-or-restore decision for a model.

    Args:
        trainer (Trainer): The trainer under test.
        model_name (str): The name of the model.

    Returns:
        Any: Whatever the trainer built or restored.
    """
    experiment_name = trainer.experiment_name(model_name)
    restore = trainer._claim_experiment(experiment_name)  # pylint: disable=protected-access
    return trainer._build_or_restore_tuner(  # pylint: disable=protected-access
        trainable=lambda config: None,
        param_space={},
        tune_config=None,
        run_config=None,
        experiment_name=experiment_name,
        restore=restore,
    )


def test_a_discarded_experiment_is_set_aside_not_restored(
    tmp_path: Path, fake_tuner: type
):
    """A model whose state was discarded starts fresh, keeping the old sweep aside."""
    old = a_ray_experiment(tmp_path, "run__EASE")
    trainer = Trainer(storage_path=str(tmp_path), run_name="run", resumable=False)

    assert build_tuner(trainer, "EASE") != "restored"
    assert fake_tuner.restored == []
    assert len(fake_tuner.built) == 1
    assert not old.exists()
    aside = list(tmp_path.glob("run__EASE.discarded-*"))
    assert len(aside) == 1
    assert (aside[0] / "tuner.pkl").read_bytes() == b"old experiment"


def test_a_resumable_experiment_is_restored(tmp_path: Path, fake_tuner: type):
    """A model the manifest says is resumable continues its own sweep."""
    old = a_ray_experiment(tmp_path, "run__BPR")
    trainer = Trainer(storage_path=str(tmp_path), run_name="run", resumable=True)

    assert build_tuner(trainer, "BPR") == "restored"
    assert fake_tuner.restored == [str(old)]
    assert old.exists()
    assert list(tmp_path.glob("*.discarded-*")) == []


def test_a_fresh_model_leaves_the_storage_alone(tmp_path: Path, fake_tuner: type):
    """Nothing is moved when there is no previous experiment."""
    trainer = Trainer(storage_path=str(tmp_path), run_name="run", resumable=False)

    build_tuner(trainer, "EASE")
    assert fake_tuner.restored == []
    assert list(tmp_path.iterdir()) == []


def test_two_discards_of_the_same_model_both_survive(tmp_path: Path, fake_tuner: type):
    """Setting aside twice in quick succession never overwrites the first one."""
    trainer = Trainer(storage_path=str(tmp_path), run_name="run", resumable=False)

    a_ray_experiment(tmp_path, "run__EASE")
    build_tuner(trainer, "EASE")
    a_ray_experiment(tmp_path, "run__EASE")
    build_tuner(trainer, "EASE")

    assert len(list(tmp_path.glob("run__EASE.discarded-*"))) == 2
    assert fake_tuner.restored == []


def test_a_new_run_resumes_nothing():
    """A model the manifest has never seen has nothing to resume."""
    state = a_run()

    model_state, resumable = reconcile_model_state(state, "EASE", PARAMS)

    assert resumable is False
    assert model_state.status is ModelStatus.PENDING
    assert model_state.fingerprint == model_fingerprint("EASE", PARAMS)


def test_an_unchanged_interrupted_model_resumes():
    """The same configuration continues where it was paused."""
    state = a_run(
        EASE=ModelState(
            status=ModelStatus.INTERRUPTED,
            fingerprint=model_fingerprint("EASE", PARAMS),
        )
    )

    model_state, resumable = reconcile_model_state(state, "EASE", PARAMS)

    assert resumable is True
    assert model_state.status is ModelStatus.INTERRUPTED


def test_a_changed_completed_model_is_optimised_again():
    """A completed model whose configuration changed is not skipped."""
    state = a_run(
        BPR=ModelState(
            status=ModelStatus.COMPLETED,
            fingerprint=model_fingerprint("BPR", PARAMS),
            best_params={"l2": 10},
        )
    )

    model_state, resumable = reconcile_model_state(state, "BPR", CHANGED)

    assert resumable is False
    assert model_state.status is ModelStatus.PENDING
    assert model_state.best_params is None
    assert state.models["BPR"] is model_state
    assert model_state.fingerprint == model_fingerprint("BPR", CHANGED)


def test_a_changed_failed_model_is_tried_again():
    """A model that failed is retried once its configuration changes."""
    state = a_run(
        BPR=ModelState(
            status=ModelStatus.FAILED, fingerprint=model_fingerprint("BPR", PARAMS)
        )
    )

    model_state, resumable = reconcile_model_state(state, "BPR", CHANGED)

    assert resumable is False
    assert model_state.status is ModelStatus.PENDING


def test_an_unchanged_completed_model_stays_completed():
    """A completed, unchanged model keeps its status and is skipped as before."""
    state = a_run(
        BPR=ModelState(
            status=ModelStatus.COMPLETED, fingerprint=model_fingerprint("BPR", PARAMS)
        )
    )

    model_state, resumable = reconcile_model_state(state, "BPR", PARAMS)

    assert resumable is True
    assert model_state.status is ModelStatus.COMPLETED
