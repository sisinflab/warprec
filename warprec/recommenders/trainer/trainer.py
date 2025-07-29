import os
import tempfile
import torch
from typing import List, Tuple, Optional, Dict, Any
from copy import deepcopy

import ray
from ray import tune
from ray.tune import Tuner, TuneConfig, RunConfig, CheckpointConfig, Checkpoint
from ray.tune.stopper import Stopper
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.air.integrations.mlflow import MLflowLoggerCallback
from codecarbon import EmissionsTracker
from warprec.recommenders.base_recommender import Recommender
from warprec.data.dataset import Dataset
from warprec.evaluation.evaluator import Evaluator
from warprec.recommenders.trainer.search_algorithm_wrapper import (
    BaseSearchWrapper,
)
from warprec.recommenders.trainer.scheduler_wrapper import (
    BaseSchedulerWrapper,
)
from warprec.utils.config import (
    Configuration,
    RecomModel,
    DashboardConfig,
    Wandb,
    CodeCarbon,
    MLflow,
)
from warprec.utils.callback import WarpRecCallback
from warprec.utils.enums import SearchSpace
from warprec.utils.logger import logger
from warprec.utils.registry import (
    model_registry,
    params_registry,
    search_algorithm_registry,
    scheduler_registry,
    search_space_registry,
)


class Trainer:
    """This class will be used to train a model and optimize the hyperparameters.

    Args:
        model_name (str): The name of the model to optimize.
        param (dict): The parameters of the model already in
            Ray Tune format.
        dataset (Dataset): The dataset to use during training.
        metric_name (str): The name of the metric that will be used
            as validation.
        top_k (int): The cutoff tu use as validation.
        beta (float): The beta value for the evaluation.
        pop_ratio (float): The pop ratio value for the evaluation.
        ray_verbose (int): The Ray level of verbosity.
        custom_callback (WarpRecCallback): The custom callback to use
            during training and evaluation. Default is an empty
            WarpRecCallback instance.
        enable_wandb (bool): Wether or not to enable Wandb.
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
        config (Configuration): The configuration of the experiment.
    """

    def __init__(
        self,
        model_name: str,
        param: dict,
        dataset: Dataset,
        metric_name: str,
        top_k: int,
        beta: float = 1.0,
        pop_ratio: float = 0.8,
        ray_verbose: int = 1,
        custom_callback: WarpRecCallback = WarpRecCallback(),
        enable_wandb: bool = False,
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
        config: Configuration = None,
    ):
        if config:
            model_params = config.models[model_name]
            dashboard = config.dashboard
        else:
            model_params = param
            dashboard = DashboardConfig(
                wandb=Wandb(
                    enabled=enable_wandb,
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

        self.infos = dataset.info()
        self._model_params: RecomModel = params_registry.get(model_name, **model_params)
        self._dashboard = dashboard
        self._custom_callback = custom_callback
        self.model_name = model_name
        self._evaluator = Evaluator(
            [metric_name],
            [top_k],
            train_set=dataset.train_set.get_sparse(),
            beta=beta,
            pop_ratio=pop_ratio,
            feature_lookup=dataset.get_features_lookup(),
            user_cluster=dataset.get_user_cluster(),
            item_cluster=dataset.get_item_cluster(),
        )
        self._train_param = self.parse_params(param)
        self._metric_name = metric_name
        self._top_k = top_k
        self._verbose = ray_verbose
        self._dataset = ray.put(dataset)

    def train_and_evaluate(self) -> Tuple[Recommender, List[Tuple[str, dict]]]:
        """Main method of the Trainer class.

        This method will execute the training of the model and evaluation,
        according to information passed through configuration.

        Returns:
            Tuple[Recommender, List[Tuple[str, dict]]]:
                - Recommender: The model trained.
                - List[Tuple[str, dict]]:
                    - str: Path of checkpoint
                    - dict: Params of the model
        """
        logger.separator()
        logger.msg(
            f"Starting hyperparameter tuning for {self.model_name} "
            f"with {self._model_params.optimization.strategy.name} strategy "
            f"and with {self._model_params.optimization.scheduler.name} scheduler."
        )

        properties = self._model_params.optimization.properties.model_dump()
        mode = self._model_params.optimization.properties.mode
        device = self._model_params.optimization.device
        keep_all_ray_checkpoints = self._model_params.meta.keep_all_ray_checkpoints

        # Ray Tune parameters
        obj_function = tune.with_parameters(
            self._objective_function,
            model_name=self.model_name,
            dataset=self._dataset,
            mode=mode,
            evaluator=self._evaluator,
            device=device,
        )

        search_alg: BaseSearchWrapper = search_algorithm_registry.get(
            self._model_params.optimization.strategy, **properties
        )
        scheduler: BaseSchedulerWrapper = scheduler_registry.get(
            self._model_params.optimization.scheduler, **properties
        )

        early_stopping: Stopper = None
        if self._model_params.early_stopping is not None:
            early_stopping = EarlyStopping(
                metric=self._model_params.early_stopping.monitor,
                mode=mode,
                patience=self._model_params.early_stopping.patience,
                grace_period=self._model_params.early_stopping.grace_period,
                min_delta=self._model_params.early_stopping.min_delta,
            )

        # Setup callbacks
        callbacks: List[tune.Callback] = [self._custom_callback]

        if self._dashboard.wandb.enabled:
            callbacks.append(
                WandbLoggerCallback(
                    project=self._dashboard.wandb.project,
                    group=self._dashboard.wandb.group,
                    api_key_file=self._dashboard.wandb.api_key_file,
                    api_key=self._dashboard.wandb.api_key,
                    excludes=self._dashboard.wandb.excludes,
                    log_config=self._dashboard.wandb.log_config,
                    upload_checkpoints=self._dashboard.wandb.upload_checkpoints,
                )
            )
        if self._dashboard.codecarbon.enabled:
            callbacks.append(
                CodeCarbonCallback(
                    save_to_api=self._dashboard.codecarbon.save_to_api,
                    save_to_file=self._dashboard.codecarbon.save_to_file,
                    output_dir=self._dashboard.codecarbon.output_dir,
                    tracking_mode=self._dashboard.codecarbon.tracking_mode,
                )
            )
        if self._dashboard.mlflow.enabled:
            callbacks.append(
                MLflowLoggerCallback(
                    tracking_uri=self._dashboard.mlflow.tracking_uri,
                    registry_uri=self._dashboard.mlflow.registry_uri,
                    experiment_name=self._dashboard.mlflow.experiment_name,
                    tags=self._dashboard.mlflow.tags,
                    tracking_token=self._dashboard.mlflow.tracking_token,
                    save_artifact=self._dashboard.mlflow.save_artifacts,
                )
            )

        # Configure Ray Tune Tuner
        run_config = RunConfig(
            stop=early_stopping,
            callbacks=callbacks,
            verbose=self._verbose,
            checkpoint_config=CheckpointConfig(
                num_to_keep=1 if not keep_all_ray_checkpoints else None,
                checkpoint_score_attribute="score",
                checkpoint_score_order=mode,
            ),
        )

        tune_config = TuneConfig(
            metric="score",
            mode=mode,
            search_alg=search_alg,  # type: ignore[arg-type]
            scheduler=scheduler,  # type: ignore[arg-type]
            num_samples=self._model_params.optimization.num_samples,
        )

        tuner = Tuner(
            tune.with_resources(
                obj_function,
                resources={
                    "cpu": self._model_params.optimization.cpu_per_trial,
                    "gpu": self._model_params.optimization.gpu_per_trial,
                },
            ),
            param_space=self._train_param,
            tune_config=tune_config,
            run_config=run_config,
        )

        # Run the hyperparameter tuning
        results = tuner.fit()

        # Retrieve results
        best_result = results.get_best_result(metric="score", mode=mode)
        best_params = best_result.config
        best_score = best_result.metrics["score"]
        best_iter = best_result.metrics["training_iteration"]
        best_checkpoint = best_result.checkpoint

        logger.msg(
            f"Best params combination: {best_params} with a score of "
            f"{self._metric_name}@"
            f"{self._top_k}: "
            f"{best_score} "
            f"during iteration {best_iter}."
        )
        logger.positive(
            f"Hyperparameter tuning for {self.model_name} ended successfully."
        )

        # Retrieve best model from checkpoint
        checkpoint_path = os.path.join(best_checkpoint.to_directory(), "checkpoint.pt")
        checkpoint_data = torch.load(checkpoint_path)
        model_state = checkpoint_data["model_state"]

        best_model = model_registry.get(
            name=self.model_name,
            implementation=self._model_params.meta.implementation,
            params=best_params,
            device=device,
            seed=self._model_params.optimization.properties.seed,
            info=self.infos,
        )
        best_model.load_state_dict(model_state)

        # Get all checkpoints and their parameters
        all_checkpoints = [
            (r.checkpoint.to_directory(), r.config)
            for r in results  # type: ignore[attr-defined]
            if r.checkpoint
        ]

        ray.shutdown()

        return best_model, all_checkpoints

    def _objective_function(
        self,
        params: dict,
        model_name: str,
        dataset: Dataset,
        mode: str,
        evaluator: Evaluator,
        device: str,
    ) -> None:
        """Objective function to optimize the hyperparameters.

        Args:
            params (dict): The parameter to train the model.
            model_name (str): The name of the model to train.
            dataset (Dataset): The dataset to train the model on.
            mode (str): Whether or not to maximize or minimize the metric.
            evaluator (Evaluator): The evaluator that will calculate the
                validation metric.
            device (str): The device used for tensor operations.
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
            score = results[key][self._top_k][self._metric_name]

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
                logger.negative(
                    str(e)
                )  # Log the custom message from Pydantic validation

                # Report to Ray Tune the trial failed
                if mode == "max":
                    tune.report(metrics={"score": -float("inf")})
                else:
                    tune.report(metrics={"score": float("inf")})

                return  # Stop Ray Tune trial

        # Proceed with normal model training behavior
        model = model_registry.get(
            name=model_name,
            implementation=self._model_params.meta.implementation,
            params=params,
            device=device,
            seed=self._model_params.optimization.properties.seed,
            info=self.infos,
            block_size=self._model_params.optimization.block_size,
        )
        try:
            model.fit(
                dataset.train_set, sessions=dataset.train_session, report_fn=_report
            )
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

    def parse_params(self, params: dict) -> dict:
        """This method parses the parameters of a model.

        From simple lists it creates the correct data format for
        Ray Tune hyperparameter optimization. The correct format depends
        on the search space desired. An example could be:
        ['uniform', 5.0, 100.0] -> tune.uniform(5.0, 100.0)

        Args:
            params (dict): The parameters of the model.

        Returns:
            dict: The parameters in the Ray Tune format.
        """
        tune_params = {}
        params_copy = deepcopy(params)
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

        return tune_params


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
