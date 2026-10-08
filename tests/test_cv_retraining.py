"""The model retrained after cross-validation trains for the summarised epochs.

Cross-validation picks, beside the hyperparameters, how many epochs the best
configuration needed on the folds (desired_training_it). The retraining on the
full training set has to train exactly that long, not for the 'epochs' upper
bound the search was given.
"""

import pytest

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.pipelines.remotes.ml import remote_model_retraining
from warprec.utils.registry import params_registry

HYPERPARAMETERS = {
    "embedding_size": 8,
    "reg_weight": 1e-5,
    "batch_size": 64,
    "epochs": 6,
    "learning_rate": 0.001,
}


@pytest.mark.parametrize("iterations", [1, 3, 6])
def test_retraining_trains_for_the_summarised_iterations(
    dataset: Dataset, iterations: int
):
    """The retrained model has seen exactly 'iterations' epochs."""
    params = params_registry.get(
        "BPR", **HYPERPARAMETERS, optimization={"num_workers": 0}
    )
    best_params = {**HYPERPARAMETERS, "iterations": iterations}

    # The function Ray wraps, run in this process
    model, _, returned = remote_model_retraining._function(  # type: ignore[attr-defined]
        model_name="BPR",
        best_params=best_params,
        main_dataset=dataset,
        params=params,
        custom_modules=[],
        device="cpu",
        seed=42,
    )

    assert returned == iterations
    assert model.trainer.current_epoch == iterations
