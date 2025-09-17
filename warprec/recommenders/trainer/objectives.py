import os
import tempfile
from typing import Any, List

import torch
import psutil
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
    metrics: List[str],
    topk: List[int],
    validation_top_k: int,
    validation_metric_name: str,
    mode: str,
    device: str,
    strategy: str = "full",
    num_negatives: int = 99,
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
        metrics (List[str]): List of metrics to compute on each report.
        topk (List[int]): List of cutoffs for metrics.
        validation_top_k (int): The number of top items to consider for evaluation.
        validation_metric_name (str): The name of the metric to optimize.
        mode (str): Whether or not to maximize or minimize the metric.
        device (str): The device used for tensor operations.
        strategy (str): Evaluation strategy, either "full" or "sampled".
            Defaults to "full".
        num_negatives (int): Number of negative samples to use in "sampled" strategy.
            Defaults to 99.
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
        metrics,
        topk,
        train_set=dataset.train_set.get_sparse(),
        beta=beta,
        pop_ratio=pop_ratio,
        feature_lookup=dataset.get_features_lookup(),
        user_cluster=dataset.get_user_cluster(),
        item_cluster=dataset.get_item_cluster(),
    )

    # Trial parameter configuration check for consistency
    model_params: RecomModel = (
        params_registry.get(model_name, **params)
        if model_name.upper() in params_registry.list_registered()
        else RecomModel(**params)
    )
    if model_params.need_single_trial_validation:
        try:
            model_params.validate_single_trial_params()
        except ValueError as e:
            logger.negative(str(e))  # Log the custom message from Pydantic validation
            failed_report(mode, validation_score)

            return  # Stop Ray Tune trial

    # Memory reporting
    process = psutil.Process(os.getpid())
    initial_ram_mb = process.memory_info().rss / 1024**2
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)

    # Proceed with normal model training behavior
    try:
        model = model_registry.get(
            name=model_name,
            params=params,
            interactions=dataset.train_set,
            device=device,
            seed=seed,
            info=dataset.info(),
            **dataset.get_stash(),
            block_size=block_size,
        )
        if isinstance(model, IterativeRecommender):
            # Proceed with standard training loop
            train_dataloader = model.get_dataloader(
                interactions=dataset.train_set, sessions=dataset.train_session
            )
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=model.learning_rate,
                weight_decay=model.weight_decay,
            )
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

                # Update memory usage
                ram_peak_mb = max(initial_ram_mb, process.memory_info().rss / 1024**2)

                # Evaluation at the end of each training epoch
                evaluator.evaluate(
                    model,
                    dataset,
                    device=device,
                    strategy=strategy,
                    num_negatives=num_negatives,
                )
                results = evaluator.compute_results()
                metric_report = {
                    f"{metric_name}@{k}": value
                    for k, metrics_results in results.items()
                    for metric_name, value in metrics_results.items()
                }
                validation_report(
                    model,
                    **metric_report,
                    loss=epoch_loss,
                    ram_peak_mb=ram_peak_mb,
                    vram_peak_mb=(
                        torch.cuda.max_memory_allocated(device=device) / 1024**2
                    )
                    if str(device) != "cpu" and torch.cuda.is_available()
                    else 0,
                )

        else:
            # Model is trained in the __init__ we can directly evaluate it
            evaluator.evaluate(
                model,
                dataset,
                device=device,
                strategy=strategy,
                num_negatives=num_negatives,
            )
            results = evaluator.compute_results()
            metric_report = {
                f"{metric_name}@{k}": value
                for k, metrics_results in results.items()
                for metric_name, value in metrics_results.items()
            }

            # Update memory usage
            ram_peak_mb = max(initial_ram_mb, process.memory_info().rss / 1024**2)

            validation_report(
                model=model,
                **metric_report,
                ram_peak_mb=ram_peak_mb,
                vram_peak_mb=(torch.cuda.max_memory_allocated(device=device) / 1024**2)
                if str(device) != "cpu" and torch.cuda.is_available()
                else 0,
            )

    except Exception as e:
        logger.negative(
            f"The fitting of the model {model.name}, failed "
            f"with parameters: {params}. Error: {e}"
        )
        failed_report(mode, validation_score)


def validation_report(model: Recommender, **kwargs: Any):
    # If the score has been computed per user we report only the mean
    for key, value in kwargs.items():
        if isinstance(value, Tensor):
            kwargs[key] = value.mean()
    with tempfile.TemporaryDirectory() as tmpdir:
        torch.save(
            {"model_state": model.state_dict()},
            os.path.join(tmpdir, "checkpoint.pt"),
        )
        tune.report(
            metrics={
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
