import os
import torch
import uuid
import math
from typing import List, Tuple, Optional, Dict, Any
from copy import deepcopy

import ray
import numpy as np
from ray import tune
from ray.tune import Tuner, TuneConfig, RunConfig, CheckpointConfig
from ray.tune.stopper import Stopper
from ray.tune.experiment import Trial
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.air.integrations.mlflow import MLflowLoggerCallback
from codecarbon import EmissionsTracker
from warprec.recommenders.base_recommender import Recommender
from warprec.data.dataset import Dataset
from warprec.recommenders.trainer.objectives import objective_function
from warprec.recommenders.trainer.search_algorithm_wrapper import (
    BaseSearchWrapper,
)
from warprec.recommenders.trainer.scheduler_wrapper import (
    BaseSchedulerWrapper,
)
from warprec.utils.config import (
    TrainConfiguration,
    RecomModel,
    DashboardConfig,
    Wandb,
    CodeCarbon,
    MLflow,
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


class Trainer:
    """This class will be used to train a model and optimize the hyperparameters.

    Args:
        custom_callback (WarpRecCallback): The custom callback to use
            during training and evaluation. Default is an empty
            WarpRecCallback instance.
        custom_models (str | List[str]): The list of custom models to load.
        enable_wandb (bool): Wether or not to enable Wandb.
        team_wandb (Optional[str]): The name of the Wandb team.
        project_wandb (str): The name of the Wandb project.
        group_wandb (Optional[str]): The name of the Wandb group.
        api_key_file_wandb (Optional[str]): The path to the Wandb
            API key file.
        api_key_wandb (Optional[str]): The Wandb API key.
        excludes_wandb (list): The list of parameters to exclude
            from Wandb logging.
        log_config_wandb (bool): Wether or not to log the config
            in Wandb.
        upload_checkpoints_wandb (bool): Wether or not to upload
            checkpoints to Wandb.
        enable_codecarbon (bool): Wether or not to enable CodeCarbon.
        save_to_api_codecarbon (bool): Wether or not to save
            CodeCarbon results to API.
        save_to_file_codecarbon (bool): Wether or not to save
            CodeCarbon results to file.
        output_dir_codecarbon (str): The directory to save
            CodeCarbon results.
        tracking_mode_codecarbon (str): The tracking mode for
            CodeCarbon. Either "machine" or "process".
        enable_mlflow (bool): Wether or not to enable MLflow.
        tracking_uri_mlflow (str): The URI of the MLflow tracking server.
        registry_uri_mlflow (str): The URI of the MLflow model registry.
        experiment_name_mlflow (Optional[str]): The name of the MLflow experiment.
        tags_mlflow (dict): The tags to be added to the MLflow run.
        tracking_token_mlflow (Optional[str]): The token for MLflow tracking.
        save_artifacts_mlflow (bool): Wether or not to save artifacts
            in MLflow.
        config (TrainConfiguration): The configuration of the experiment.
    """

    def __init__(
        self,
        custom_callback: WarpRecCallback = WarpRecCallback(),
        custom_models: str | List[str] = [],
        enable_wandb: bool = False,
        team_wandb: Optional[str] = None,
        project_wandb: str = "WarpRec",
        group_wandb: Optional[str] = None,
        api_key_file_wandb: Optional[str] = None,
        api_key_wandb: Optional[str] = None,
        excludes_wandb: list = [],
        log_config_wandb: bool = False,
        upload_checkpoints_wandb: bool = False,
        enable_codecarbon: bool = False,
        save_to_api_codecarbon: bool = False,
        save_to_file_codecarbon: bool = False,
        output_dir_codecarbon: str = "./",
        tracking_mode_codecarbon: str = "machine",
        enable_mlflow: bool = False,
        tracking_uri_mlflow: str = "mlruns/",
        registry_uri_mlflow: str = "mlruns/",
        experiment_name_mlflow: Optional[str] = None,
        tags_mlflow: dict = {},
        tracking_token_mlflow: Optional[str] = None,
        save_artifacts_mlflow: bool = False,
        config: TrainConfiguration = None,
    ):
        if config:
            dashboard = config.dashboard
        else:
            dashboard = DashboardConfig(
                wandb=Wandb(
                    enabled=enable_wandb,
                    team=team_wandb,
                    project=project_wandb,
                    group=group_wandb,
                    api_key_file=api_key_file_wandb,
                    api_key=api_key_wandb,
                    excludes=excludes_wandb,
                    log_config=log_config_wandb,
                    upload_checkpoints=upload_checkpoints_wandb,
                ),
                codecarbon=CodeCarbon(
                    enabled=enable_codecarbon,
                    save_to_api=save_to_api_codecarbon,
                    save_to_file=save_to_file_codecarbon,
                    output_dir=output_dir_codecarbon,
                    tracking_mode=tracking_mode_codecarbon,
                ),
                mlflow=MLflow(
                    enabled=enable_mlflow,
                    tracking_uri=tracking_uri_mlflow,
                    registry_uri=registry_uri_mlflow,
                    experiment_name=experiment_name_mlflow,
                    tags=tags_mlflow,
                    tracking_token=tracking_token_mlflow,
                    save_artifacts=save_artifacts_mlflow,
                ),
            )

        self._callbacks = self._setup_callbacks(dashboard, custom_callback)
        self._custom_models = custom_models

    def train_single_fold(
        self,
        model_name: str,
        params: RecomModel,
        dataset: Dataset,
        metrics: List[str],
        topk: List[int],
        validation_score: str,
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
        # Retrieve common parameters
        mode = params.optimization.properties.mode
        seed = params.optimization.properties.seed

        # Prepare the Tuner
        tuner = self._prepare_trainable(
            model_name=model_name,
            params=params,
            dataset=dataset,
            metrics=metrics,
            topk=topk,
            validation_score=validation_score,
            device=device,
            evaluation_strategy=evaluation_strategy,
            num_negatives=num_negatives,
            beta=beta,
            pop_ratio=pop_ratio,
            ray_verbose=ray_verbose,
        )

        # Run the hyperparameter tuning
        results = tuner.fit()

        # Retrieve results
        best_result = results.get_best_result(metric=validation_score, mode=mode)

        # Early check for no successful trials
        if (
            mode == "max"
            and best_result.metrics[validation_score] == -torch.inf
            or mode == "min"
            and best_result.metrics[validation_score] == torch.inf
        ):
            logger.negative(
                f"All trials failed during training for {model_name}. Shutting down the Trainer."
            )
            ray.shutdown()
            return None, {}
        best_params = best_result.config
        best_score = best_result.metrics[validation_score]
        best_iter = best_result.metrics["training_iteration"]
        best_checkpoint = best_result.checkpoint

        # Memory report
        result_df = results.get_dataframe()
        if "ram_peak_mb" in result_df.columns and "vram_peak_mb" in result_df.columns:
            additional_report = {
                "RAM Mean Usage (MB)": result_df["ram_peak_mb"].mean(),
                "RAM STD Usage (MB)": result_df["ram_peak_mb"].std(),
                "RAM Max Usage (MB)": result_df["ram_peak_mb"].max(),
                "RAM Min Usage (MB)": result_df["ram_peak_mb"].min(),
                "VRAM Mean Usage (MB)": result_df["vram_peak_mb"].mean(),
                "VRAM STD Usage (MB)": result_df["vram_peak_mb"].std(),
                "VRAM Max Usage (MB)": result_df["vram_peak_mb"].max(),
                "VRAM Min Usage (MB)": result_df["vram_peak_mb"].min(),
            }
        else:
            additional_report = {}

        logger.msg(
            f"Best params combination: {best_params} with a score of "
            f"{validation_score}: {best_score} "
            f"during iteration {best_iter}."
        )
        logger.positive(f"Hyperparameter tuning for {model_name} ended successfully.")

        # Retrieve best model from checkpoint
        checkpoint_path = os.path.join(best_checkpoint.to_directory(), "checkpoint.pt")
        checkpoint_data = torch.load(checkpoint_path, weights_only=True)
        model_state = checkpoint_data["model_state"]

        best_model = model_registry.get(
            name=model_name,
            params=best_params,
            interactions=dataset.train_set,
            device=device,
            seed=seed,
            info=dataset.info(),
            **dataset.get_stash(),
        )
        best_model.load_state_dict(model_state)

        report = self._create_report(results, additional_report, best_model)

        ray.shutdown()

        return best_model, report

    def train_multiple_fold(
        self,
        model_name: str,
        params: RecomModel,
        datasets: List[Dataset],
        metrics: List[str],
        topk: List[int],
        validation_score: str,
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
        # Retrieve common parameters
        mode = params.optimization.properties.mode

        # Prepare the Tuner
        tuner = self._prepare_trainable(
            model_name=model_name,
            params=params,
            dataset=datasets,
            metrics=metrics,
            topk=topk,
            validation_score=validation_score,
            device=device,
            evaluation_strategy=evaluation_strategy,
            num_negatives=num_negatives,
            beta=beta,
            pop_ratio=pop_ratio,
            ray_verbose=ray_verbose,
        )

        # Run the hyperparameter tuning
        results = tuner.fit()

        # Find the hyperparameter configuration that performed better
        result_df = results.get_dataframe(
            filter_metric=validation_score,
            filter_mode=mode,
        )

        # Early check for no successful trials
        if (
            mode == "max"
            and result_df[validation_score].max() == -torch.inf
            or mode == "min"
            and result_df[validation_score].min() == torch.inf
        ):
            logger.negative(
                f"All trials failed during training for {model_name}. Shutting down the Trainer."
            )
            ray.shutdown()
            return None, {}

        hyperparam_cols = [
            col
            for col in result_df.columns
            if col.startswith("config/") and col != "config/fold"
        ]

        # WarpRec params with be treated as lists, we need
        # to convert them to tuple in order to hash them
        for col in hyperparam_cols:
            if col in result_df.columns and result_df[col].dtype == "object":
                result_df[col] = result_df[col].apply(
                    lambda x: tuple(x) if isinstance(x, list) else x
                )

        # Aggregate results over hyperparameter combinations and compute mean and std
        agg_df = (
            result_df.groupby(hyperparam_cols)
            .agg(
                mean_score=(validation_score, "mean"),
                std_score=(validation_score, "std"),
                num_folds_completed=(validation_score, "size"),
                desired_training_iterations=("training_iteration", desired_training_it),
            )
            .reset_index()
        )

        # Order by mean to find best hyperparameters (ordering will be dependent on mode)
        best_config_df = agg_df.sort_values(
            by="mean_score", ascending=True if mode == "min" else False
        )
        best_hyperparameters_row = best_config_df.iloc[0]
        best_mean_score = best_hyperparameters_row["mean_score"]
        best_std_score = best_hyperparameters_row["std_score"]
        desired_iteration = math.ceil(
            best_hyperparameters_row["desired_training_iterations"]
        )

        # Memory report
        additional_report = {
            "RAM Mean Usage (MB)": result_df["ram_peak_mb"].mean(),
            "RAM STD Usage (MB)": result_df["ram_peak_mb"].std(),
            "RAM Max Usage (MB)": result_df["ram_peak_mb"].max(),
            "RAM Min Usage (MB)": result_df["ram_peak_mb"].min(),
            "VRAM Mean Usage (MB)": result_df["vram_peak_mb"].mean(),
            "VRAM STD Usage (MB)": result_df["vram_peak_mb"].std(),
            "VRAM Max Usage (MB)": result_df["vram_peak_mb"].max(),
            "VRAM Min Usage (MB)": result_df["vram_peak_mb"].min(),
        }

        # Clear hyperparam format and create the clean dictionary
        best_hyperparameters: Dict[str, Any] = {}
        best_hyperparameters["iterations"] = desired_iteration
        for col in hyperparam_cols:
            param_name = col.replace("config/", "")
            value = best_hyperparameters_row[col]

            if isinstance(value, np.floating) and value == int(value):
                # This check converts aggregated hyperparameters that
                # can become floating values back to integers
                best_hyperparameters[param_name] = int(value)
            elif isinstance(value, np.integer):
                best_hyperparameters[param_name] = int(value)
            elif isinstance(value, np.floating):
                best_hyperparameters[param_name] = float(value)
            elif isinstance(value, np.bool_):
                best_hyperparameters[param_name] = bool(value)
            else:
                best_hyperparameters[param_name] = value

        logger.msg(
            f"Best params combination: {best_hyperparameters} with an average score of "
            f"{validation_score}: {best_mean_score} and "
            f"STD: {best_std_score} on validation set. "
            f"The {desired_training_it} of training iteration is: {desired_iteration}"
        )
        logger.positive(f"Hyperparameter tuning for {model_name} ended successfully.")

        report = self._create_report(results, additional_report)

        ray.shutdown()

        return best_hyperparameters, report

    def parse_params(self, params: RecomModel, num_folds: int = 0) -> dict:
        """This method parses the parameters of a model.

        From simple lists it creates the correct data format for
        Ray Tune hyperparameter optimization. The correct format depends
        on the search space desired. An example could be:
        ['uniform', 5.0, 100.0] -> tune.uniform(5.0, 100.0)

        Args:
            params (RecomModel): The parameters of the model.
            num_folds (int): The number of cross-validation folds.

        Returns:
            dict: The parameters in the Ray Tune format.
        """
        tune_params = {}
        params_copy = deepcopy(params.model_dump())
        if "meta" in params_copy:
            params_copy.pop("meta")
        if "optimization" in params_copy:
            params_copy.pop("optimization")
        if "early_stopping" in params_copy:
            params_copy.pop("early_stopping")

        for k, v in params_copy.items():
            if v[0] is not SearchSpace.CHOICE:
                tune_params[k] = search_space_registry.get(v[0])(*v[1:])
            else:
                tune_params[k] = search_space_registry.get(v[0])(v[1:])

        if num_folds > 0:
            tune_params["fold"] = tune.grid_search(list(range(num_folds)))

        return tune_params

    def trail_name(self, model_name: str):
        def _trial_name_creator(trial: Trial):
            random_id = str(uuid.uuid4())[:8]

            return f"{model_name}_{random_id}"

        return _trial_name_creator

    def _prepare_trainable(
        self,
        model_name: str,
        params: RecomModel,
        dataset: Dataset | List[Dataset],
        metrics: List[str],
        topk: List[int],
        validation_score: str,
        device: str = "cpu",
        evaluation_strategy: str = "full",
        num_negatives: int = 99,
        beta: float = 1.0,
        pop_ratio: float = 0.8,
        ray_verbose: int = 1,
    ) -> Tuner:
        # Retrieve common information
        properties = params.optimization.properties.model_dump()
        optimization = params.optimization
        mode = params.optimization.properties.mode
        validation_metric_name, validation_top_k = validation_metric(validation_score)

        # Log the start of HPO setup
        logger.separator()
        num_folds = 0
        if isinstance(dataset, list):
            num_folds = len(dataset)
            logger.msg(
                f"Starting hyperparameter tuning for {model_name} "
                f"with {optimization.strategy.name} strategy "
                f"and with {optimization.scheduler.name} scheduler. "
                f"Number of validation folds: {num_folds}"
            )
        else:
            logger.msg(
                f"Starting hyperparameter tuning for {model_name} "
                f"with {optimization.strategy.name} strategy "
                f"and with {optimization.scheduler.name} scheduler."
            )

        # Ray Tune parameters
        obj_function = tune.with_parameters(
            objective_function,
            model_name=model_name,
            dataset_folds=ray.put(dataset),
            metrics=metrics,
            topk=topk,
            validation_top_k=validation_top_k,
            validation_metric_name=validation_metric_name,
            mode=mode,
            device=device,
            strategy=evaluation_strategy,
            num_negatives=num_negatives,
            seed=optimization.properties.seed,
            block_size=optimization.block_size,
            beta=beta,
            pop_ratio=pop_ratio,
            custom_models=self._custom_models,
        )

        search_alg: BaseSearchWrapper = search_algorithm_registry.get(
            optimization.strategy, **properties
        )
        scheduler: BaseSchedulerWrapper = scheduler_registry.get(
            optimization.scheduler, **properties
        )

        early_stopping: Stopper = None
        if params.early_stopping is not None:
            early_stopping = EarlyStopping(
                metric=validation_score,
                mode=mode,
                patience=params.early_stopping.patience,
                grace_period=params.early_stopping.grace_period,
                min_delta=params.early_stopping.min_delta,
            )

        # Configure Ray Tune Tuner
        run_config = RunConfig(
            stop=early_stopping,
            callbacks=self._callbacks,
            verbose=ray_verbose,
            checkpoint_config=CheckpointConfig(
                num_to_keep=optimization.checkpoint_to_keep,
                checkpoint_score_attribute=validation_score,
                checkpoint_score_order=mode,
            ),
        )

        tune_config = TuneConfig(
            metric=validation_score,
            mode=mode,
            search_alg=search_alg,  # type: ignore[arg-type]
            scheduler=scheduler,  # type: ignore[arg-type]
            num_samples=optimization.num_samples,
            trial_name_creator=self.trail_name(model_name),
        )

        tuner = Tuner(
            tune.with_resources(
                obj_function,
                resources={
                    "cpu": optimization.max_cpu_count // optimization.parallel_trials,
                    "gpu": min(
                        torch.cuda.device_count() / optimization.parallel_trials, 1.0
                    ),
                },
            ),
            param_space=self.parse_params(params, num_folds),
            tune_config=tune_config,
            run_config=run_config,
        )

        return tuner

    def _setup_callbacks(
        self, dashboard: DashboardConfig, custom_callback: WarpRecCallback
    ) -> List[tune.Callback]:
        callbacks: List[tune.Callback] = [custom_callback]

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
                    entity=dashboard.wandb.team,  # Will be passed to wandb.init()
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
        self,
        results: tune.ResultGrid,
        additional_reports: Dict[str, float],
        model: Optional[Recommender] = None,
    ) -> dict:
        # Produce the report of the training
        successful_trials = [r for r in results if not r.error]  # type: ignore[attr-defined]
        report = {}
        if successful_trials:
            total_trial_times = [r.metrics["time_total_s"] for r in successful_trials]
            report["Average Trial Time"] = sum(total_trial_times) / len(
                total_trial_times
            )

        if model is not None:
            report["Total Params (Best Model)"] = sum(
                p.numel() for p in model.parameters()
            )
            report["Trainable Params (Best Model)"] = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

        # Add additional reports
        report.update(additional_reports)

        return report


class CodeCarbonCallback(tune.Callback):
    def __init__(
        self,
        save_to_api: bool = False,
        save_to_file: bool = False,
        output_dir: str = "./",
        tracking_mode: str = "machine",
    ):
        self.trackers: Dict[str, EmissionsTracker] = {}
        self.save_to_api = save_to_api
        self.save_to_file = save_to_file
        self.output_dir = output_dir
        self.tracking_mode = tracking_mode

    def on_trial_start(self, iteration, trials, trial, **info):
        tracker = EmissionsTracker(
            save_to_api=self.save_to_api,
            save_to_file=self.save_to_file,
            output_dir=self.output_dir,
            tracking_mode=self.tracking_mode,
        )
        tracker.start()
        self.trackers[trial.trial_id] = tracker

    def on_trial_complete(self, iteration, trials, trial, **info):
        tracker = self.trackers.pop(trial.trial_id, None)
        if tracker:
            tracker.stop()

    def on_trial_fail(self, iteration, trials, trial, **info):
        tracker = self.trackers.pop(trial.trial_id, None)
        if tracker:
            tracker.stop()


class EarlyStopping(Stopper):
    """Ray Tune Stopper for early stopping based on a validation metric.

    Args:
        metric (str): The name of the metric to monitor for early stopping.
        mode (str): One of {"min", "max"}. In "min" mode, training will stop
            when the quantity monitored has stopped decreasing; in "max" mode
            it will stop when the quantity monitored has stopped increasing.
        patience (int): Number of epochs with no improvement after which
            training will be stopped.
        grace_period (int): Number of epochs to wait before activating
            the stopper.
        min_delta (float): Minimum change in the monitored quantity to qualify
            as an improvement, i.e. an absolute change of less than min_delta,
            will count as no improvement.

    Raises:
        ValueError: If the mode is not 'min' or 'max'.
    """

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
        self.trial_best_score: Dict[str, Optional[float]] = {}
        self.trial_wait: Dict[str, int] = {}

    def __call__(self, trial_id: str, result: Dict) -> bool:
        """Callback when a trial reports a result.

        Args:
            trial_id (str): The id of the trial.
            result (Dict): The result dictionary.

        Returns:
            bool: Wether or not to suppress the trial.
        """

        current_score = result.get(self.metric, None)
        iteration = result.get("training_iteration", None)

        if current_score is None:
            logger.attention(
                f"Metric '{self.metric}' not found in trial results for trial {trial_id}. "
                "Early stopping will not be applied to this trial in this iteration."
            )
            return False

        if trial_id not in self.trial_best_score:
            self.trial_best_score[trial_id] = current_score
            self.trial_wait[trial_id] = 0
        elif iteration <= self.grace_period:
            return False
        else:
            if self.mode == "min":
                if current_score < self.trial_best_score[trial_id] - self.min_delta:
                    self.trial_best_score[trial_id] = current_score
                    self.trial_wait[trial_id] = 0
                else:
                    self.trial_wait[trial_id] += 1
            elif self.mode == "max":
                if current_score > self.trial_best_score[trial_id] + self.min_delta:
                    self.trial_best_score[trial_id] = current_score
                    self.trial_wait[trial_id] = 0
                else:
                    self.trial_wait[trial_id] += 1

        if self.trial_wait[trial_id] >= self.patience:
            logger.attention(
                f"Early stopping triggered for trial {trial_id}: "
                f"No improvement in '{self.metric}' for {self.patience} iterations. "
            )
            return True
        return False

    def stop_all(self):
        return False
