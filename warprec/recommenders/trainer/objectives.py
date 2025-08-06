import os
import tempfile
from typing import Any, List

import torch
from torch import Tensor
from ray import tune
from ray.tune import Checkpoint

from warprec.data.dataset import Dataset
from warprec.evaluation.evaluator import Evaluator
from warprec.recommenders.base_recommender import Recommender, IterativeRecommender
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
            failed_report(mode, validation_score)

            return  # Stop Ray Tune trial

    # Proceed with normal model training behavior
    try:
        model = model_registry.get(
            name=model_name,
            params=params,
            interactions=dataset.train_set,
            device=device,
            seed=seed,
            info=dataset.info(),
            block_size=block_size,
        )
        if isinstance(model, IterativeRecommender):
            # Proceed with standard training loop
            train_dataloader = model.get_dataloader(dataset.train_set)
            optimizer = model.get_optimizer()
            epochs = model.epochs

            model.train()
            for epoch in range(epochs):
                epoch_loss = 0.0
                for batch in train_dataloader:
                    optimizer.zero_grad()

                    loss = model.train_step(batch, epoch)
                    loss.backward()

                    optimizer.step()
                    epoch_loss += loss.item()

                # Evaluation at the end of each training epoch
                evaluator.evaluate(model, dataset, device=device)
                results = evaluator.compute_results()
                score = results[validation_top_k][validation_metric_name]
                validation_report(model, validation_score, score, loss=epoch_loss)

        else:
            # Model is trained in the __init__ we can directly evaluate it
            evaluator.evaluate(model, dataset, device=device)
            results = evaluator.compute_results()
            score = results[validation_top_k][validation_metric_name]
            validation_report(model, validation_score, score)

    except Exception as e:
        logger.negative(
            f"The fitting of the model {model.name}, failed "
            f"with parameters: {params}. Error: {e}"
        )
        failed_report(mode, validation_score)


def validation_report(
    model: Recommender, validation_score: str, score: float | Tensor, **kwargs: Any
):
    # If the score has been computed per user we report only the mean
    report_score = score if isinstance(score, float) else score.mean()
    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(
            {"model_state": model.state_dict()},
            os.path.join(tmpdir, "checkpoint.pt"),
        )
        tune.report(
            metrics={
                validation_score: report_score,
                **kwargs,
            },
            checkpoint=Checkpoint.from_directory(tmpdir),
        )


def failed_report(mode: str, validation_score: str):
    # Report to Ray Tune the trial failed
    if mode == "max":
        tune.report(
            metrics={validation_score: -float("inf")},
        )
    else:
        tune.report(
            metrics={validation_score: float("inf")},
        )
