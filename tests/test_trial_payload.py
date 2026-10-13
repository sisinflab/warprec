"""What the train pipeline ships to every Ray Tune trial.

The trial function is pickled once per trial and sent to the process that runs
it. It used to read the storage path off the Trainer, so the closure captured
the whole Trainer, and with it the custom callback and the dashboard callbacks:
every trial unpickled its own copy of each, running their unpickling code and
constructors' side effects in the trial process, though only the driver calls
them.
"""

from pathlib import Path
from typing import Any, Dict

import pytest
from ray import cloudpickle

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.trainer import trainer as trainer_module
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config.evaluation_configuration import EvaluationConfig
from warprec.utils.registry import params_registry


class CountingCallback(WarpRecCallback):
    """A custom callback that counts how many times it is unpickled."""

    unpickled = 0

    def __init__(self) -> None:
        super().__init__()
        self.scores: list = []

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Count one more copy of the callback.

        Args:
            state (Dict[str, Any]): The pickled attributes.
        """
        CountingCallback.unpickled += 1
        self.__dict__.update(state)


def test_a_trial_does_not_carry_the_callbacks(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset, tmp_path: Path
):
    """The trial function and its parameters pickle without any callback."""
    shipped: Dict[str, Any] = {}
    monkeypatch.setattr(trainer_module.ray, "put", lambda value: value)
    monkeypatch.setattr(
        trainer_module.tune,
        "with_parameters",
        lambda fn, **kwargs: shipped.update(fn=fn, **kwargs) or fn,
    )
    monkeypatch.setattr(trainer_module.tune, "with_resources", lambda fn, resources: fn)
    monkeypatch.setattr(
        trainer_module.Trainer, "_build_or_restore_tuner", lambda self, **kw: None
    )

    params = params_registry.get(
        "BPR",
        embedding_size=8,
        reg_weight=1e-5,
        batch_size=64,
        epochs=1,
        learning_rate=0.001,
        optimization={"max_concurrent_trials": 1},
    )
    trainer = trainer_module.Trainer(
        storage_path=str(tmp_path), custom_callback=CountingCallback()
    )
    trainer._setup_tuner(
        model_name="BPR",
        params=params,
        dataset=dataset,
        evaluation=EvaluationConfig(
            top_k=[5], metrics=["nDCG"], validation_metric="nDCG@5"
        ),
        validation_score="nDCG@5",
        device="cpu",
        ray_verbose=0,
    )

    CountingCallback.unpickled = 0
    cloudpickle.loads(cloudpickle.dumps(shipped))

    assert CountingCallback.unpickled == 0
