import os
import tempfile
import types
from typing import Any, List, Dict


import torch
import torch.distributed as dist
import psutil
from torch import Tensor

from ray import tune, train
from ray.tune import Checkpoint
from ray.tune.integration.ray_train import TuneReportCallback
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer, get_device

from warprec.data import Dataset
from warprec.evaluation.evaluator import Evaluator
from warprec.recommenders.base_recommender import Recommender, IterativeRecommender
from warprec.utils.config import RecomModel
from warprec.utils.helpers import load_custom_modules, retrieve_evaluation_dataloader
from warprec.utils.registry import model_registry, params_registry
from warprec.utils.logger import logger


def _get_memory_report(
    process: psutil.Process,
    initial_ram_mb: float,
    device: torch.device | str,
) -> Dict[str, float]:
    """Calculates and returns a dictionary with peak RAM and VRAM usage.

    Args:
        process (psutil.Process): The process to track.
        initial_ram_mb (float): The initial ram value.
        device (torch.device | str): The device of the experiment.

    Returns:
        Dict[str, float]: The memory report.
    """
    ram_peak_mb = max(initial_ram_mb, process.memory_info().rss / 1024**2)
    vram_peak_mb = 0.0
    if str(device) != "cpu" and torch.cuda.is_available():
        vram_peak_mb = torch.cuda.max_memory_allocated(device=device) / 1024**2

    return {"ram_peak_mb": ram_peak_mb, "vram_peak_mb": vram_peak_mb}


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
    # Memory reporting
    process = psutil.Process(os.getpid())
    initial_ram_mb = process.memory_info().rss / 1024**2
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)

    # Validation metric in the correct format
    validation_score = f"{validation_metric_name}@{validation_top_k}"

    # Load custom modules if provided
    load_custom_modules(custom_models)

    # Extract the correct dataset
    if isinstance(dataset_folds, list):
        fold_index: int = params["fold"]
        dataset = dataset_folds[fold_index]
    else:
        dataset = dataset_folds

    # Retrieve appropriate evaluation dataloader
    dataloader = retrieve_evaluation_dataloader(
        dataset=dataset,
        strategy=strategy,
        num_negatives=num_negatives,
    )

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
            failed_report(mode, validation_score, "tune")

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

            for epoch in range(epochs):
                # Set model to train mode at the beginning of each epoch
                model.train()
                epoch_loss = 0.0
                for batch in train_dataloader:
                    optimizer.zero_grad()

                    loss = model.train_step(batch, epoch)
                    loss.backward()

                    optimizer.step()
                    epoch_loss += loss.item()

                # Set model to eval mode for the evaluation step
                model.eval()

                # Evaluation at the end of each training epoch
                evaluator.evaluate(
                    model=model,
                    dataloader=dataloader,
                    strategy=strategy,
                    dataset=dataset,
                    device=device,
                )
                results = evaluator.compute_results()

                # Metrics to report
                metric_report = {
                    f"{metric_name}@{k}": value
                    for k, metrics_results in results.items()
                    for metric_name, value in metrics_results.items()
                }
                metric_report["loss"] = epoch_loss / len(train_dataloader)

                memory_report = _get_memory_report(process, initial_ram_mb, device)

                validation_report(
                    model=model,
                    **metric_report,
                    **memory_report,
                )

        else:
            # Model is trained in the __init__ we can directly evaluate it
            evaluator.evaluate(
                model=model,
                dataloader=dataloader,
                strategy=strategy,
                dataset=dataset,
                device=device,
            )
            results = evaluator.compute_results()

            # Metrics to report
            metric_report = {
                f"{metric_name}@{k}": value
                for k, metrics_results in results.items()
                for metric_name, value in metrics_results.items()
            }
            memory_report = _get_memory_report(process, initial_ram_mb, device)

            validation_report(
                model=model,
                **metric_report,
                **memory_report,
            )

    except Exception as e:
        logger.negative(
            f"The fitting of the model {model_name}, failed "
            f"with parameters: {params}. Error: {e}"
        )
        failed_report(mode, validation_score, "tune")


def objective_function_ddp(config: dict) -> None:
    # Monitor memory consumption
    process = psutil.Process(os.getpid())
    initial_ram_mb = process.memory_info().rss / 1024**2
    device = get_device()
    if str(device) != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)

    # Load custom modules if provided
    custom_models = config["custom_models"]
    load_custom_modules(custom_models)

    # Initialize the model and dataset
    model_name = config["model_name"]
    params = config["params"]
    dataset_folds = config["dataset_folds"]
    mode = config["mode"]
    validation_metric_name = config["validation_metric_name"]
    validation_top_k = config["validation_top_k"]
    device = get_device()

    # Validation metric in the correct format
    validation_score = f"{validation_metric_name}@{validation_top_k}"

    # Define world size for metric reporting
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Trial validation on rank 0
    if train.get_context().get_world_rank() == 0:
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
                logger.negative(str(e))  # Log error message
                failed_report(mode, validation_score, "train")
                return  # Interrupt this worker

    # Extract the correct dataset for this fold
    dataset: Dataset
    if isinstance(dataset_folds, list):
        fold_index = params["fold"]
        dataset = dataset_folds[fold_index]
    else:
        dataset = dataset_folds

    # Instantiate the model with hyperparameters
    model = model_registry.get(
        name=model_name,
        params=config["params"],
        interactions=dataset.train_set,
        device=device,
        seed=config["seed"],
        info=dataset.info(),
        **dataset.get_stash(),
        block_size=config["block_size"],
    )

    # Only IterativeRecommender models are supported in DDP
    if not isinstance(model, IterativeRecommender):
        raise ValueError(
            "The DDP objective function only supports IterativeRecommender models."
        )

    model = train.torch.prepare_model(model)  # Wrap model for DDP training
    unwrapped_model = model.module  # Used for correct typing

    if not isinstance(unwrapped_model, IterativeRecommender):
        raise ValueError("Something went wrong.")

    # Prepare the distributed train dataloader
    train_dataloader = unwrapped_model.get_dataloader(
        interactions=dataset.train_set, sessions=dataset.train_session
    )
    train_dataloader = train.torch.prepare_data_loader(train_dataloader)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        unwrapped_model.parameters(),
        lr=unwrapped_model.learning_rate,
        weight_decay=unwrapped_model.weight_decay,
    )

    # Prepare the evaluation dataloader
    eval_dataloader = retrieve_evaluation_dataloader(
        dataset=dataset,
        strategy=config["strategy"],
        num_negatives=config["num_negatives"],
    )
    eval_dataloader = train.torch.prepare_data_loader(eval_dataloader)

    # Initialize the Evaluator
    evaluator = Evaluator(
        config["metrics"],
        config["topk"],
        train_set=dataset.train_set.get_sparse(),
        beta=config["beta"],
        pop_ratio=config["pop_ratio"],
        feature_lookup=dataset.get_features_lookup(),
        user_cluster=dataset.get_user_cluster(),
        item_cluster=dataset.get_item_cluster(),
    )

    # Training loop
    for epoch in range(unwrapped_model.epochs):
        # Set model to train mode at the beginning of each epoch
        model.train()
        epoch_loss = 0.0

        # Shuffle data loading with DistributedSampler
        train_dataloader.sampler.set_epoch(epoch)  # type:ignore [attr-defined]

        for batch in train_dataloader:
            optimizer.zero_grad()

            loss = unwrapped_model.train_step(batch, epoch)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()

        # Set model to eval mode for the evaluation step
        model.eval()

        # Evaluation step distributed across all workers
        evaluator.evaluate(
            model=unwrapped_model,
            dataloader=eval_dataloader,
            strategy=config["strategy"],
            dataset=dataset,
            device=device,
        )

        # All workers call compute(). torchmetrics handles synchronization.
        # The `results` dictionary will be identical on all workers.
        results = evaluator.compute_results()
        metric_report = {
            f"{metric_name}@{k}": value
            for k, metrics_results in results.items()
            for metric_name, value in metrics_results.items()
        }

        # Add the loss averaged over the train sample size
        metric_report["loss"] = epoch_loss / (len(train_dataloader) * world_size)

        # Memory reporting on rank 0
        if train.get_context().get_world_rank() == 0:
            memory_report = _get_memory_report(process, initial_ram_mb, device)
            metric_report.update(memory_report)

        # Report metrics and checkpoint from all workers
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint only on rank 0 to prevent race conditions
            if train.get_context().get_world_rank() == 0:
                torch.save(
                    {"model_state": unwrapped_model.state_dict()},
                    os.path.join(tmpdir, "checkpoint.pt"),
                )
            # Ensure rank 0 has saved before other ranks try to access it
            dist.barrier()

            train.report(
                metrics=metric_report,
                checkpoint=tune.Checkpoint.from_directory(tmpdir),
            )

        # Synchronize all processes before the next epoch
        dist.barrier()


def driver_function_ddp(
    params: dict,
    model_name: str,
    dataset_folds: Dataset | List[Dataset],
    metrics: List[str],
    topk: List[int],
    validation_top_k: int,
    validation_metric_name: str,
    mode: str,
    num_gpus: int,
    storage_path: str,
    num_to_keep: int = 5,
    strategy: str = "full",
    num_negatives: int = 99,
    seed: int = 42,
    block_size: int = 50,
    beta: float = 1.0,
    pop_ratio: float = 0.8,
    custom_models: List[str] = [],
):
    trainer = TorchTrainer(
        objective_function_ddp,
        train_loop_config={
            "params": params,
            "model_name": model_name,
            "dataset_folds": dataset_folds,
            "metrics": metrics,
            "topk": topk,
            "validation_top_k": validation_top_k,
            "validation_metric_name": validation_metric_name,
            "mode": mode,
            "strategy": strategy,
            "num_negatives": num_negatives,
            "seed": seed,
            "block_size": block_size,
            "beta": beta,
            "pop_ratio": pop_ratio,
            "custom_models": custom_models,
        },
        scaling_config=ScalingConfig(
            num_workers=num_gpus,
            use_gpu=True,
        ),
        run_config=RunConfig(
            callbacks=[TuneReportCallback()],  # type: ignore[list-item]
            storage_path=storage_path,
            checkpoint_config=CheckpointConfig(
                num_to_keep=num_to_keep,
                checkpoint_score_attribute=f"{validation_metric_name}@{validation_top_k}",
                checkpoint_score_order=mode,
            ),
        ),
    )
    trainer.fit()


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


def failed_report(mode: str, validation_score: str, report_type: str):
    # Define reporting API
    reporter: types.ModuleType
    if report_type == "tune":
        reporter = tune
    elif report_type == "train":
        reporter = train
    else:
        raise ValueError("Report type not valid.")

    # Report to correct APU the failed trial
    if mode == "max":
        reporter.report(
            metrics={validation_score: -float("inf")},
        )
    else:
        reporter.report(
            metrics={validation_score: float("inf")},
        )
