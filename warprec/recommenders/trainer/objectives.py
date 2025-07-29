import os
import tempfile
from typing import Any

import torch
from ray import tune
from ray.tune import Checkpoint

from warprec.data.dataset import Dataset
from warprec.evaluation.evaluator import Evaluator
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.config import RecomModel
from warprec.utils.registry import model_registry, params_registry
from warprec.utils.logger import logger


def objective_function(
    params: dict,
    model_name: str,
    dataset: Dataset,
    info: dict,
    top_k: int,
    metric_name: str,
    mode: str,
    evaluator: Evaluator,
    device: str,
    implementation: str = "latest",
    seed: int = 42,
    block_size: int = 50,
) -> None:
    """Objective function to optimize the hyperparameters.

    Args:
        params (dict): The parameter to train the model.
        model_name (str): The name of the model to train.
        dataset (Dataset): The dataset to train the model on.
        info (dict): Additional information about the dataset.
        top_k (int): The number of top items to consider for evaluation.
        metric_name (str): The name of the metric to optimize.
        mode (str): Whether or not to maximize or minimize the metric.
        evaluator (Evaluator): The evaluator that will calculate the
            validation metric.
        device (str): The device used for tensor operations.
        implementation (str): The implementation of the model to use.
            Defaults to "latest".
        seed (int): The seed for reproducibility. Defaults to 42.
        block_size (int): The block size for the model optimization.
            Defaults to 50.

    Returns:
        None: This function reports metrics and checkpoints to Ray Tune
            via `tune.report()` and does not explicitly return a value.
    """

    def _report(model: Recommender, **kwargs: Any):
        """Reporting function. Will be used as a callback for Tune reporting.

        Args:
            model (Recommender): The trained model to report.
            **kwargs (Any): The parameters of the model.
        """
        key: str
        if dataset.val_set is not None:
            key = "validation"
            evaluator.evaluate(model, dataset, evaluate_on_validation=True)
        else:
            key = "test"
            evaluator.evaluate(model, dataset, evaluate_on_test=True)

        results = evaluator.compute_results()
        score = results[key][top_k][metric_name]

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                {"model_state": model.state_dict()},
                os.path.join(tmpdir, "checkpoint.pt"),
            )
            tune.report(
                metrics={
                    "score": score,
                    **kwargs,  # Other metrics from the model itself
                },
                checkpoint=Checkpoint.from_directory(tmpdir),
            )

    # Trial parameter configuration check for consistency
    model_params: RecomModel = params_registry.get(model_name, **params)
    if model_params.need_single_trial_validation:
        try:
            model_params.validate_single_trial_params()
        except ValueError as e:
            logger.negative(str(e))  # Log the custom message from Pydantic validation

            # Report to Ray Tune the trial failed
            if mode == "max":
                tune.report(metrics={"score": -float("inf")})
            else:
                tune.report(metrics={"score": float("inf")})

            return  # Stop Ray Tune trial

    # Proceed with normal model training behavior
    model = model_registry.get(
        name=model_name,
        implementation=implementation,
        params=params,
        device=device,
        seed=seed,
        info=info,
        block_size=block_size,
    )
    try:
        model.fit(dataset.train_set, sessions=dataset.train_session, report_fn=_report)
    except Exception as e:
        logger.negative(
            f"The fitting of the model {model.name}, failed "
            f"with parameters: {params}. Error: {e}"
        )
        if mode == "max":
            tune.report(
                metrics={"score": -torch.inf},
            )
        else:
            tune.report(
                metrics={"score": torch.inf},
            )
