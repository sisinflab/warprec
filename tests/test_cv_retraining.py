"""The model retrained after cross-validation trains for the summarised epochs.

Cross-validation picks, beside the hyperparameters, how many epochs the best
configuration needed on the folds (desired_training_it). The retraining on the
full training set has to train exactly that long, not for the 'epochs' upper
bound the search was given.
"""

import pandas as pd
import pytest

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.pipelines.remotes.ml import remote_model_retraining
from warprec.recommenders.trainer.trainer import Trainer
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


def test_the_retrained_model_describes_the_epochs_it_trained(dataset: Dataset):
    """The parameters saved with the model give the epochs it was trained for."""
    params = params_registry.get(
        "BPR", **HYPERPARAMETERS, optimization={"num_workers": 0}
    )

    model, _, _ = remote_model_retraining._function(  # type: ignore[attr-defined]
        model_name="BPR",
        best_params={**HYPERPARAMETERS, "iterations": 2},
        main_dataset=dataset,
        params=params,
        custom_modules=[],
        device="cpu",
        seed=42,
    )

    assert model.get_params()["epochs"] == 2
    assert model.get_state()["params"]["epochs"] == 2


def test_cross_validation_keeps_a_float_hyperparameter_a_float():
    """A float that happens to be whole stays a float in the best parameters."""
    frame = pd.DataFrame(
        {
            "config/l2": [500.0, 500.0, 10.5, 10.5],
            "config/k": [10, 10, 20, 20],
            "config/fold": [0, 1, 0, 1],
            "nDCG@10": [0.3, 0.32, 0.1, 0.12],
            "iterations": [1, 1, 1, 1],
        }
    )

    best, _ = Trainer(storage_path="unused")._aggregate_cv_results(
        frame, "nDCG@10", "max", "median"
    )

    assert best["l2"] == 500.0 and isinstance(best["l2"], float)
    assert best["k"] == 10 and isinstance(best["k"], int)
