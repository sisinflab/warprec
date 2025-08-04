import os
import tempfile
from typing import Any, List

import torch
from ray import tune
from ray.tune import Checkpoint

from warprec.data.dataset import Dataset
from warprec.evaluation.evaluator import Evaluator
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.config import RecomModel
from warprec.utils.helpers import load_custom_modules
from warprec.utils.registry import model_registry, params_registry
from warprec.utils.logger import logger


def objective_function(
    params: dict,
    model_name: str,
    dataset_folds: Dataset | List[Dataset],
    validation_top_k: int,
    validation_metric_name: str,
    mode: str,
    device: str,
    implementation: str = "latest",
    seed: int = 42,
    block_size: int = 50,
    beta: float = 1.0,
    pop_ratio: float = 0.8,
    custom_models: List[str] = [],
) -> None:
    """Objective function to optimize the hyperparameters.

    Args:
        params (dict): The parameter to train the model.
        model_name (str): The name of the model to train.
        dataset_folds (Dataset | List[Dataset]): The dataset to train the model on.
            If a list is passed, then it will be handled as folding.
        validation_top_k (int): The number of top items to consider for evaluation.
        validation_metric_name (str): The name of the metric to optimize.
        mode (str): Whether or not to maximize or minimize the metric.
        device (str): The device used for tensor operations.
        implementation (str): The implementation of the model to use.
            Defaults to "latest".
        seed (int): The seed for reproducibility. Defaults to 42.
        block_size (int): The block size for the model optimization.
            Defaults to 50.
        beta (float): The beta value to initialize the Evaluator.
        pop_ratio (float): The pop_ratio value to initialize the Evaluator.
        custom_models (List[str]): List of custom models to import.
            Defaults to an empty list.

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
        evaluator.evaluate(model, dataset, device=device)
        results = evaluator.compute_results()
        score = results[validation_top_k][validation_metric_name]

        with tempfile.TemporaryDirectory() as tmpdir:
            torch.save(
                {"model_state": model.state_dict()},
                os.path.join(tmpdir, "checkpoint.pt"),
            )
            tune.report(
                metrics={
                    validation_score: score,
                    **kwargs,  # Other metrics from the model itself
                },
                checkpoint=Checkpoint.from_directory(tmpdir),
            )

    validation_score = f"{validation_metric_name}@{validation_top_k}"

    # Load custom modules if provided
    load_custom_modules(custom_models)

    # Extract the correct dataset
    if isinstance(dataset_folds, list):
        fold_index: int = params["fold"]
        dataset = dataset_folds[fold_index]
    else:
        dataset = dataset_folds

    # Initialize the Evaluator for current Trial
    evaluator = Evaluator(
        [validation_metric_name],
        [validation_top_k],
        train_set=dataset.train_set.get_sparse(),
        beta=beta,
        pop_ratio=pop_ratio,
        feature_lookup=dataset.get_features_lookup(),
        user_cluster=dataset.get_user_cluster(),
        item_cluster=dataset.get_item_cluster(),
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
                tune.report(metrics={validation_score: -float("inf")})
            else:
                tune.report(metrics={validation_score: float("inf")})

            return  # Stop Ray Tune trial

    # Proceed with normal model training behavior
    model = model_registry.get(
        name=model_name,
        implementation=implementation,
        params=params,
        device=device,
        seed=seed,
        info=dataset.info(),
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
                metrics={validation_score: -torch.inf},
            )
        else:
            tune.report(
                metrics={validation_score: torch.inf},
            )
