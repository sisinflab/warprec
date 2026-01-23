import os
import uuid
import math
from typing import List, Tuple, Optional, Dict, Callable, Union, Any
from pathlib import Path

import torch
import numpy as np
import ray
from ray import tune
from ray.tune import Tuner, TuneConfig, CheckpointConfig, RunConfig
from ray.tune.stopper import Stopper
from ray.tune.experiment import Trial

from warprec.recommenders.base_recommender import Recommender
from warprec.data import Dataset
from warprec.recommenders.trainer.objectives import (
    objective_function,
    driver_function_ddp,
)
from warprec.utils.config import (
    TrainConfiguration,
    RecomModel,
    DashboardConfig,
)
from warprec.utils.helpers import validation_metric
from warprec.utils.callback import WarpRecCallback
from warprec.utils.enums import SearchSpace
from warprec.utils.logger import logger
from warprec.utils.registry import (
    model_registry,
    search_algorithm_registry,
    scheduler_registry,
    search_space_registry,
)

# Optional imports handling
try:
    from ray.air.integrations.wandb import WandbLoggerCallback
    from ray.air.integrations.mlflow import MLflowLoggerCallback
    from codecarbon import EmissionsTracker

    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


class Trainer:
    """Trainer class for training and hyperparameter optimization using Ray Tune.
    Delegates configuration details to TrainConfiguration object.

    Args:
        config (TrainConfiguration): The complete configuration object.
        custom_callback (WarpRecCallback): Custom callback for training/eval.
        custom_models (Optional[Union[str, List[str]]]): List of custom models to load.
    """

    def __init__(
        self,
        config: TrainConfiguration,
        custom_callback: WarpRecCallback = WarpRecCallback(),
        custom_models: Optional[Union[str, List[str]]] = None,
    ):
        self.config = config
        self._custom_models = custom_models or []
        self._callbacks = self._setup_callbacks(config.dashboard, custom_callback)

    def train_single_fold(
        self,
        model_name: str,
        params: RecomModel,
        dataset: Dataset,
        metrics: List[str],
        topk: List[int],
        validation_score: str,
        storage_path: str,
        num_gpus: Optional[int] = None,
        device: str = "cpu",
        evaluation_strategy: str = "full",
        num_negatives: int = 99,
        beta: float = 1.0,
        pop_ratio: float = 0.8,
        ray_verbose: int = 1,
    ) -> Tuple[Optional[Recommender], dict]:
        """Main method of the Trainer class.

        This method will execute the training of the model and evaluation,
        according to information passed through configuration.

        Args:
            model_name (str): The name of the model to optimize.
            params (RecomModel): The parameters of the model.
            dataset (Dataset): The dataset to use during training.
            metrics (List[str]): List of metrics to compute on each report.
            topk (List[int]): List of cutoffs for metrics.
            validation_score (str): The metric to monitor during training.
            storage_path (str): Path to store Ray results.
            num_gpus (Optional[int]): The number of gpus per trial.
            device (str): The device that will be used for tensor operations.
            evaluation_strategy (str): Evaluation strategy, either "full" or "sampled".
            num_negatives (int): Number of negative samples to use in "sampled" strategy.
            beta (float): The beta value for the evaluation.
            pop_ratio (float): The pop ratio value for the evaluation.
            ray_verbose (int): The Ray level of verbosity.

        Returns:
            Tuple[Optional[Recommender], dict]:
                - Recommender: The model trained.
                - dict: Summary report of the training.
        """

        mode = params.optimization.properties.mode

        tuner = self._setup_tuner(
            model_name=model_name,
            params=params,
            dataset=dataset,
            metrics=metrics,
            topk=topk,
            validation_score=validation_score,
            storage_path=storage_path,
            num_gpus=num_gpus,
            device=device,
            evaluation_strategy=evaluation_strategy,
            num_negatives=num_negatives,
            beta=beta,
            pop_ratio=pop_ratio,
            ray_verbose=ray_verbose,
        )

        results = tuner.fit()

        # Check if any trial succeeded
        if results.errors and len(results) == len(results.errors):
            logger.negative(f"All trials failed for {model_name}.")
            return None, {}

        # Retrieve best result using Ray API
        best_result = results.get_best_result(metric=validation_score, mode=mode)

        if not best_result:
            logger.negative(f"Could not determine best result for {model_name}.")
            return None, {}

        best_params = best_result.config
        # Remove internal ray config keys if present
        best_params = {k: v for k, v in best_params.items() if not k.startswith("_")}
        best_score = best_result.metrics.get(validation_score)
        best_iter = best_result.metrics.get("training_iteration")

        logger.msg(
            f"Best params: {best_params} | Score ({validation_score}): {best_score} "
            f"| Iteration: {best_iter}"
        )
        logger.positive(f"HPO for {model_name} ended successfully.")

        # Load Best Model
        best_model = self._load_best_model(
            model_name,
            best_result,
            best_params,
            dataset,
            device,
            params.optimization.properties.seed,
        )

        report = self._create_report(results, best_model)
        return best_model, report

    def train_multiple_fold(
        self,
        model_name: str,
        params: RecomModel,
        datasets: List[Dataset],
        metrics: List[str],
        topk: List[int],
        validation_score: str,
        storage_path: str,
        num_gpus: Optional[int] = None,
        device: str = "cpu",
        evaluation_strategy: str = "full",
        num_negatives: int = 99,
        beta: float = 1.0,
        pop_ratio: float = 0.8,
        desired_training_it: str = "median",
        ray_verbose: int = 1,
    ) -> Tuple[Optional[Dict], Dict]:
        """Main method of the Trainer class for cross-validation.

        Args:
            model_name (str): The name of the model to optimize.
            params (RecomModel): The parameters of the model.
            datasets (List[Dataset]): The list of datasets to use during training.
            metrics (List[str]): List of metrics to compute on each report.
            topk (List[int]): List of cutoffs for metrics.
            validation_score (str): The metric to monitor during training.
            storage_path (str): Path to store Ray results.
            num_gpus (Optional[int]): The number of gpus per trial.
            device (str): The device that will be used for tensor operations.
            evaluation_strategy (str): Evaluation strategy, either "full" or "sampled".
            num_negatives (int): Number of negative samples to use in "sampled" strategy.
            beta (float): The beta value for the evaluation.
            pop_ratio (float): The pop ratio value for the evaluation.
            desired_training_it (str): The type of statistic to use to
                select the number of training iterations to use
                when training on the full dataset. Either "min", "max",
                "mean" or "median". Default is "median".
            ray_verbose (int): The Ray level of verbosity.

        Returns:
            Tuple[Optional[Dict], Dict]:
                - Dict: The best hyperparameters found.
                - Dict: Summary report of the training.
        """

        mode = params.optimization.properties.mode

        tuner = self._setup_tuner(
            model_name=model_name,
            params=params,
            dataset=datasets,
            metrics=metrics,
            topk=topk,
            validation_score=validation_score,
            storage_path=storage_path,
            num_gpus=num_gpus,
            device=device,
            evaluation_strategy=evaluation_strategy,
            num_negatives=num_negatives,
            beta=beta,
            pop_ratio=pop_ratio,
            ray_verbose=ray_verbose,
        )

        results = tuner.fit()
        result_df = results.get_dataframe(
            filter_metric=validation_score, filter_mode=mode
        )

        if result_df.empty or (
            mode == "max" and result_df[validation_score].max() == -torch.inf
        ):
            logger.negative(f"All trials failed for {model_name}.")
            return None, {}

        # Aggregate results logic
        best_hyperparameters, best_stats = self._aggregate_cv_results(
            result_df, validation_score, mode, desired_training_it
        )

        logger.msg(
            f"Best params: {best_hyperparameters} | Avg Score: {best_stats['mean']} "
            f"| Std: {best_stats['std']} | Iterations: {best_hyperparameters['iterations']}"
        )
        logger.positive(f"CV HPO for {model_name} ended successfully.")

        report = self._create_report(results)
        return best_hyperparameters, report

    def _setup_tuner(
        self,
        model_name: str,
        params: RecomModel,
        dataset: Union[Dataset, List[Dataset]],
        metrics: List[str],
        topk: List[int],
        validation_score: str,
        storage_path: str,
        num_gpus: Optional[int],
        device: str,
        evaluation_strategy: str,
        num_negatives: int,
        beta: float,
        pop_ratio: float,
        ray_verbose: int,
    ) -> Tuner:
        """Prepares the Ray Tuner instance.

        Args:
            model_name (str): The name of the model to optimize.
            params (RecomModel): The parameters of the model.
            dataset (Union[Dataset, List[Dataset]]): The dataset(s) to use during
                training.
            metrics (List[str]): List of metrics to compute on each report.
            topk (List[int]): List of cutoffs for metrics.
            validation_score (str): The metric to monitor during training.
            storage_path (str): Path to store Ray results.
            num_gpus (Optional[int]): The number of gpus per trial.
            device (str): The device that will be used for tensor operations.
            evaluation_strategy (str): Evaluation strategy, either "full" or "sampled".
            num_negatives (int): Number of negative samples to use in "sampled" strategy.
            beta (float): The beta value for the evaluation.
            pop_ratio (float): The pop ratio value for the evaluation.
            ray_verbose (int): The Ray level of verbosity.

        Returns:
            Tuner: The configured Ray Tuner instance.
        """

        opt_config = params.optimization
        mode = opt_config.properties.mode

        # Determine resources and objective function
        resources = self._get_resources(opt_config.parallel_trials, num_gpus)
        trainable = self._get_objective_function(
            model_name=model_name,
            params=params,
            dataset=dataset,
            metrics=metrics,
            topk=topk,
            validation_score=validation_score,
            storage_path=storage_path,
            num_gpus=num_gpus,
            device=device,
            evaluation_strategy=evaluation_strategy,
            num_negatives=num_negatives,
            beta=beta,
            pop_ratio=pop_ratio,
            resources=resources,
        )

        # Search Algorithm & Scheduler
        search_alg = search_algorithm_registry.get(
            opt_config.strategy, **opt_config.properties.model_dump()
        )
        scheduler = scheduler_registry.get(
            opt_config.scheduler, **opt_config.properties.model_dump()
        )

        # Early Stopping
        stopper = None
        if params.early_stopping:
            stopper = EarlyStopping(
                metric=validation_score,
                mode=mode,
                patience=params.early_stopping.patience,
                grace_period=params.early_stopping.grace_period,
                min_delta=params.early_stopping.min_delta,
            )

        # Configs
        run_config = RunConfig(
            stop=stopper,
            callbacks=self._callbacks,
            verbose=ray_verbose,
            storage_path=storage_path,
            checkpoint_config=CheckpointConfig(
                num_to_keep=opt_config.checkpoint_to_keep,
                checkpoint_score_attribute=validation_score,
                checkpoint_score_order=mode,
            ),
        )

        tune_config = TuneConfig(
            metric=validation_score,
            mode=mode,
            search_alg=search_alg,  # type: ignore[arg-type]
            scheduler=scheduler,  # type: ignore[arg-type]
            num_samples=opt_config.num_samples,
            trial_name_creator=self._trial_name_creator(model_name),
        )

        num_folds = len(dataset) if isinstance(dataset, list) else 0
        param_space = self._parse_params(params, num_folds)

        return Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )

    def _get_resources(
        self, num_trials: int, num_gpus: Optional[int]
    ) -> Dict[str, float]:
        """Calculates resource allocation per trial.

        Args:
            num_trials (int): The number of parallel trials.
            num_gpus (Optional[int]): The number of gpus per trial.

        Returns:
            Dict[str, float]: Resources dictionary for Ray Tune.
        """
        # Get available resources
        resources = ray.available_resources()
        available_cpus = resources.get("CPU", 0)
        available_gpus = resources.get("GPU", 0)

        if num_gpus is not None and num_gpus > 1:
            return {"cpu": available_cpus}

        return {
            "cpu": available_cpus / num_trials,
            "gpu": min(available_gpus / num_trials, 1.0) if num_gpus else 0.0,
        }

    def _get_objective_function(self, **kwargs: Any) -> Callable:
        """Selects and wraps the appropriate objective function (Standard or DDP).

        Args:
            **kwargs (Any): Keyword arguments for the objective functions.

        Returns:
            Callable: The wrapped objective function.
        """
        num_gpus = kwargs.get("num_gpus")
        params = kwargs.get("params")
        opt_config = params.optimization
        validation_metric_name, validation_top_k = validation_metric(
            kwargs["validation_score"]
        )

        common_args = {
            "model_name": kwargs["model_name"],
            "dataset_folds": ray.put(kwargs["dataset"]),
            "metrics": kwargs["metrics"],
            "topk": kwargs["topk"],
            "validation_top_k": validation_top_k,
            "validation_metric_name": validation_metric_name,
            "mode": opt_config.properties.mode,
            "low_memory": params.meta.low_memory,
            "strategy": kwargs["evaluation_strategy"],
            "num_negatives": kwargs["num_negatives"],
            "lr_scheduler": opt_config.lr_scheduler,
            "seed": opt_config.properties.seed,
            "block_size": opt_config.block_size,
            "chunk_size": opt_config.chunk_size,
            "beta": kwargs["beta"],
            "pop_ratio": kwargs["pop_ratio"],
            "custom_models": self._custom_models,
        }

        if num_gpus is not None and num_gpus > 1:
            logger.msg(f"Using Distributed Data Parallel with {num_gpus} GPUs.")
            obj_func = tune.with_parameters(
                driver_function_ddp,
                num_gpus=num_gpus,
                storage_path=kwargs["storage_path"],
                num_to_keep=opt_config.checkpoint_to_keep,
                **common_args,
            )
        else:
            obj_func = tune.with_parameters(
                objective_function, device=kwargs["device"], **common_args
            )

        return tune.with_resources(obj_func, resources=kwargs["resources"])

    def _load_best_model(
        self, model_name, best_result, best_params, dataset, device, seed
    ):
        """Loads the model state from the best checkpoint."""
        checkpoint = best_result.checkpoint
        if not checkpoint:
            # Fallback if checkpoint object is missing but path exists in metrics
            ckpt_path = (
                Path(best_result.metrics.get("checkpoint_path", "")) / "checkpoint.pt"
            )
        else:
            ckpt_path = Path(checkpoint.to_directory()) / "checkpoint.pt"

        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found at {ckpt_path}")
            return None

        # Load checkpoint on CPU, move model later
        checkpoint_data = torch.load(ckpt_path, weights_only=True, map_location="cpu")

        # Retrieve state dict
        if "model_state" in checkpoint_data:
            state_dict = checkpoint_data["model_state"]
        elif "state_dict" in checkpoint_data:
            state_dict = checkpoint_data["state_dict"]
        else:
            state_dict = checkpoint_data

        # Initialize the model and load checkpoint
        model = model_registry.get(
            name=model_name,
            params=best_params,
            interactions=dataset.train_set,
            device=device,
            seed=seed,
            info=dataset.info(),
            **dataset.get_stash(),
        )
        model.load_state_dict(state_dict)
        return model

    def _aggregate_cv_results(self, df, metric, mode, desired_it_stat):
        """Aggregates Cross-Validation results to find best hyperparameters."""
        hyperparam_cols = [
            c for c in df.columns if c.startswith("config/") and c != "config/fold"
        ]

        # Fix list hashing for groupby
        for col in hyperparam_cols:
            if df[col].dtype == "object":
                df[col] = df[col].apply(
                    lambda x: tuple(x) if isinstance(x, list) else x
                )

        agg_df = (
            df.groupby(hyperparam_cols)
            .agg(
                mean_score=(metric, "mean"),
                std_score=(metric, "std"),
                desired_training_iterations=("training_iteration", desired_it_stat),
            )
            .reset_index()
        )

        best_row = agg_df.sort_values(by="mean_score", ascending=(mode == "min")).iloc[
            0
        ]

        # Reconstruct clean params dict
        best_params = {"iterations": math.ceil(best_row["desired_training_iterations"])}
        for col in hyperparam_cols:
            key = col.replace("config/", "")
            val = best_row[col]
            # Type restoration logic
            if isinstance(val, (np.integer, int)):
                best_params[key] = int(val)
            elif isinstance(val, (np.floating, float)):
                best_params[key] = int(val) if val.is_integer() else float(val)
            elif isinstance(val, (np.bool_, bool)):
                best_params[key] = bool(val)
            else:
                best_params[key] = val

        stats = {"mean": best_row["mean_score"], "std": best_row["std_score"]}
        return best_params, stats

    def _parse_params(self, params: RecomModel, num_folds: int = 0) -> dict:
        """Parses model parameters into Ray Tune search space."""
        tune_params = {}
        # Exclude metadata fields
        exclude = {"meta", "optimization", "early_stopping"}
        clean_params = {
            k: v for k, v in params.model_dump().items() if k not in exclude
        }

        for k, v in clean_params.items():
            if isinstance(v, list) and len(v) > 0:
                space_type = v[0]
                args = v[1:]
                if space_type == SearchSpace.CHOICE:
                    tune_params[k] = search_space_registry.get(space_type)(args)
                else:
                    tune_params[k] = search_space_registry.get(space_type)(*args)
            else:
                tune_params[k] = v  # Static parameter

        if num_folds > 0:
            tune_params["fold"] = tune.grid_search(list(range(num_folds)))

        return tune_params

    def _trial_name_creator(self, model_name: str):
        def _creator(trial: Trial):
            return f"{model_name}_{str(uuid.uuid4())[:8]}"

        return _creator

    def _setup_callbacks(
        self, dashboard: DashboardConfig, custom_callback: WarpRecCallback
    ) -> List[tune.Callback | WarpRecCallback]:
        callbacks: List[tune.Callback | WarpRecCallback] = [custom_callback]
        if not DASHBOARD_AVAILABLE:
            if any(
                [
                    dashboard.wandb.enabled,
                    dashboard.codecarbon.enabled,
                    dashboard.mlflow.enabled,
                ]
            ):
                logger.attention(
                    "WarpRec dashboard extra has not been installed. "
                    "Dashboards will not be available during training."
                )
            return callbacks

        if dashboard.wandb.enabled:
            callbacks.append(
                WandbLoggerCallback(
                    project=dashboard.wandb.project,
                    group=dashboard.wandb.group,
                    api_key_file=dashboard.wandb.api_key_file,
                    api_key=dashboard.wandb.api_key,
                    excludes=dashboard.wandb.excludes,
                    log_config=dashboard.wandb.log_config,
                    upload_checkpoints=dashboard.wandb.upload_checkpoints,
                    entity=dashboard.wandb.team,
                )
            )
        if dashboard.codecarbon.enabled:
            callbacks.append(
                CodeCarbonCallback(
                    save_to_api=dashboard.codecarbon.save_to_api,
                    save_to_file=dashboard.codecarbon.save_to_file,
                    output_dir=dashboard.codecarbon.output_dir,
                    tracking_mode=dashboard.codecarbon.tracking_mode,
                )
            )
        if dashboard.mlflow.enabled:
            callbacks.append(
                MLflowLoggerCallback(
                    tracking_uri=dashboard.mlflow.tracking_uri,
                    registry_uri=dashboard.mlflow.registry_uri,
                    experiment_name=dashboard.mlflow.experiment_name,
                    tags=dashboard.mlflow.tags,
                    tracking_token=dashboard.mlflow.tracking_token,
                    save_artifact=dashboard.mlflow.save_artifacts,
                )
            )
        return callbacks

    def _create_report(
        self, results: tune.ResultGrid, model: Optional[Recommender] = None
    ) -> dict:
        report = {}

        # Memory stats from dataframe
        df = results.get_dataframe()
        mem_cols = ["ram_peak_mb", "vram_peak_mb"]
        for col in mem_cols:
            if col in df.columns:
                prefix = "RAM" if "ram" in col and "vram" not in col else "VRAM"
                report[f"{prefix} Mean Usage (MB)"] = df[col].mean()
                report[f"{prefix} Max Usage (MB)"] = df[col].max()

        # Time stats
        successful_trials = [r for r in results if not r.error]  # type: ignore[attr-defined]
        if successful_trials:
            times = [r.metrics.get("time_total_s", 0) for r in successful_trials]
            report["Average Trial Time (s)"] = sum(times) / len(times)

        # Model stats
        report["Total Params (Best Model)"] = np.nan
        report["Trainable Params (Best Model)"] = np.nan

        if model:
            report["Total Params (Best Model)"] = sum(
                p.numel() for p in model.parameters()
            )
            report["Trainable Params (Best Model)"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

        return report


class CodeCarbonCallback(tune.Callback):
    """Custom CodeCarbon callback for Ray Tune."""

    def __init__(
        self,
        save_to_api=False,
        save_to_file=False,
        output_dir="./",
        tracking_mode="machine",
    ):
        self.save_to_api = save_to_api
        self.save_to_file = save_to_file
        self.output_dir = output_dir
        self.tracking_mode = tracking_mode
        self.trackers: Dict[str, EmissionsTracker] = {}
        os.makedirs(self.output_dir, exist_ok=True)

    def on_trial_start(self, iteration, trials, trial, **info):
        tracker = EmissionsTracker(
            save_to_api=self.save_to_api,
            save_to_file=self.save_to_file,
            output_dir=self.output_dir,
            tracking_mode=self.tracking_mode,
            log_level="error",  # Reduce noise
        )
        tracker.start()
        self.trackers[trial.trial_id] = tracker

    def on_trial_complete(self, iteration, trials, trial, **info):
        self._stop_tracker(trial.trial_id)

    def on_trial_fail(self, iteration, trials, trial, **info):
        self._stop_tracker(trial.trial_id)

    def _stop_tracker(self, trial_id):
        tracker = self.trackers.pop(trial_id, None)
        if tracker:
            tracker.stop()


class EarlyStopping(Stopper):
    """Ray Tune Stopper for early stopping based on a validation metric."""

    def __init__(
        self,
        metric: str,
        mode: str,
        patience: int,
        grace_period: int = 0,
        min_delta: float = 0.0,
    ):
        if mode not in ["min", "max"]:
            raise ValueError("Mode must be 'min' or 'max'.")
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.grace_period = grace_period
        self.min_delta = min_delta
        self.trial_state: Dict[str, Dict] = {}  # Stores best_score and wait_count

    def __call__(self, trial_id: str, result: Dict) -> bool:
        score = result.get(self.metric)
        iteration = result.get("training_iteration", 0)

        if score is None:
            return False

        if trial_id not in self.trial_state:
            self.trial_state[trial_id] = {"best": score, "wait": 0}
            return False

        if iteration <= self.grace_period:
            return False

        state = self.trial_state[trial_id]
        improved = (
            (score < state["best"] - self.min_delta)
            if self.mode == "min"
            else (score > state["best"] + self.min_delta)
        )

        if improved:
            state["best"] = score
            state["wait"] = 0
        else:
            state["wait"] += 1

        if state["wait"] >= self.patience:
            logger.attention(f"Early stopping trial {trial_id} at iter {iteration}.")
            return True
        return False

    def stop_all(self):
        return False
