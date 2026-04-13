import os
import logging
import warnings
from typing import Union, Any, List

import lightning as L
import ray
from ray import train
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)
from torch import Tensor

from warprec.evaluation.evaluator import Evaluator
from warprec.recommenders.callbacks import (
    WarpRecLightningIntegrationCallback,
    _get_memory_usage,
)
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.config import RecomModel
from warprec.utils.helpers import load_custom_modules, retrieve_evaluation_dataloader
from warprec.utils.registry import (
    model_registry,
    params_registry,
)
from warprec.utils.logger import logger


def objective_function(config: dict) -> None:
    """Objective function to optimize the hyperparameters.

    Args:
        config (dict): A single dictionary containing all the training configurations
            and hyperparameters injected by Ray Tune.

    Returns:
        None: This function reports metrics and checkpoints to Ray Tune
            via `train.report()` and does not explicitly return a value.
    """
    # Setting up logging
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore", message=".*GPU available but not used.*", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message=".*sync_dist=True.*", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=UserWarning,
    )

    # Extract configurations from the config dictionary
    params = config.get("params", {})
    model_name = config["model_name"]
    dataset_folds = config["dataset_folds"]
    metrics = config["metrics"]
    topk = config["topk"]
    validation_top_k = config["validation_top_k"]
    validation_metric_name = config["validation_metric_name"]
    mode = config["mode"]
    device = config["device"]
    num_workers = config.get("num_workers", None)
    eval_every_n = config.get("eval_every_n", 1)
    strategy = config.get("strategy", "full")
    num_negatives = config.get("num_negatives", 99)
    complex_metrics = config.get("complex_metrics", None)
    lr_scheduler_config = config.get("lr_scheduler", None)
    optimizer_config = config.get("optimizer", None)
    seed = config.get("seed", 42)
    block_size = config.get("block_size", 50)
    chunk_size = config.get("chunk_size", 4096)
    custom_modules = config.get("custom_modules", [])
    early_stopping_config = config.get("early_stopping_config", None)

    # Validation metric in the correct format
    validation_score = f"{validation_metric_name}@{validation_top_k}"

    # Load custom modules if provided
    load_custom_modules(custom_modules)

    # Extract the correct dataset
    dataset_folds = ray.get(dataset_folds)
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
        additional_data=dataset.get_stash(),
        complex_metrics=complex_metrics,
        feature_lookup=dataset.get_features_lookup(),
        user_cluster=dataset.get_user_cluster(),
        item_cluster=dataset.get_item_cluster(),
    )

    # Initialize WarpRec + Lightning integration callback
    integration_callback = WarpRecLightningIntegrationCallback(
        evaluator=evaluator,
        dataset=dataset,
        strategy=strategy,
        early_stopping_config=early_stopping_config,
        validation_score=validation_score,
        mode=mode,
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
            # Report failure to Ray Train
            if train.get_context().get_world_rank() == 0:
                train.report(
                    {validation_score: -float("inf") if mode == "max" else float("inf")}
                )
            return

    # Proceed with normal model training behavior
    try:
        model = model_registry.get(
            name=model_name,
            params=params,
            interactions=dataset.train_set,
            seed=seed,
            info=dataset.info(),
            **dataset.get_stash(),
            block_size=block_size,
            chunk_size=chunk_size,
        )

        # Retrieve appropriate evaluation dataloader
        eval_dataloader = retrieve_evaluation_dataloader(
            dataset=dataset,
            model=model,
            strategy=strategy,
            num_negatives=num_negatives,
        )

        if isinstance(model, IterativeRecommender):
            # Set up the learning rate scheduler and the optimizer
            model.set_optimization_parameters(
                optimizer_config=optimizer_config,
                lr_scheduler_config=lr_scheduler_config,
            )

            # Dataloader workers logic
            if num_workers is None:
                try:
                    resources = train.get_context().get_trial_resources()
                    allocated_cpus = int(resources.get("CPU", 1))
                except Exception:
                    allocated_cpus = os.cpu_count() or 1
                num_workers = max(allocated_cpus - 1, 1)

            persistent_workers = num_workers > 0
            pin_memory = device == "cuda"

            # Proceed with standard training loop
            train_dataloader = model.get_dataloader(
                interactions=dataset.train_set,
                sessions=dataset.train_session,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
            epochs = model.epochs

            # Set the environment for the training
            l_device = device if device != "cuda" else "gpu"
            world_size = train.get_context().get_world_size()

            pl_strategy: Union[str, RayDDPStrategy]
            pl_plugins: List[Any]
            if world_size > 1:
                pl_strategy = RayDDPStrategy()
                pl_plugins = [RayLightningEnvironment()]
            else:
                pl_strategy = "auto"
                pl_plugins = []

            # Start the training process with PyTorch Lightning
            trainer = L.Trainer(
                max_epochs=epochs,
                devices="auto",
                accelerator=l_device,
                strategy=pl_strategy,  # Ray handles DDP communication
                plugins=pl_plugins,  # Ray handles environment variables
                num_sanity_val_steps=0,
                logger=False,
                enable_checkpointing=False,  # Handled by our custom callback
                enable_model_summary=False,
                enable_progress_bar=False,
                check_val_every_n_epoch=eval_every_n,
                callbacks=[integration_callback],
            )
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=eval_dataloader,
            )

        else:
            # Model is trained in the __init__ we can directly evaluate it
            evaluator.evaluate(
                model=model,
                dataloader=eval_dataloader,
                strategy=strategy,
                dataset=dataset,
                device=device,
            )
            results = evaluator.compute_results()

            # Metrics to report
            if train.get_context().get_world_rank() == 0:
                metric_report = {
                    f"{metric_name}@{k}": value.nanmean().item()
                    if isinstance(value, Tensor)
                    else value
                    for k, metrics_results in results.items()
                    for metric_name, value in metrics_results.items()
                }
                metric_report.update(_get_memory_usage())

                # Report to Ray Tune
                train.report(metrics=metric_report)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.negative(
            f"The fitting of the model {model_name}, failed "
            f"with parameters: {params}. Error: {e}"
        )
        if train.get_context().get_world_rank() == 0:
            train.report(
                {validation_score: -float("inf") if mode == "max" else float("inf")}
            )
