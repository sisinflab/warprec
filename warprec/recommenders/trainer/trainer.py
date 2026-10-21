import os

# Set Ray environment variable to enable new features
os.environ["RAY_TRAIN_V2_ENABLED"] = "1"
import hashlib
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Union, Any
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pyarrow.fs
import ray
from ray import cloudpickle, tune
from ray.tune import Tuner, TuneConfig
from ray.tune.experiment import Trial
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.integration.ray_train import TuneReportCallback
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.train._internal.storage import get_fs_and_path

from warprec.data.stash import MODEL_ARGUMENTS, check_stash
from warprec.recommenders.base_recommender import Recommender, IterativeRecommender
from warprec.data import Dataset
from warprec.recommenders.trainer.logger_callbacks import (
    WarpRecCSVLoggerCallback,
    WarpRecJsonLoggerCallback,
    read_logged_epochs,
)
from warprec.recommenders.trainer.objectives import TRIAL_ERROR, objective_function
from warprec.recommenders.trainer.search_algorithm_wrapper import (
    CappedVariantGenerator,
    ConcurrencyCap,
    ValidSuggestions,
    recap_restored_search,
)
from warprec.recommenders.trainer.timing import (
    SEARCH_OVERHEAD,
    TRIAL_STARTED_AT,
    TrialSpans,
    search_overhead,
    summarise_timing,
    trial_timing,
)
from warprec.utils.config import (
    EvaluationConfig,
    RecomModel,
    DashboardConfig,
)
from warprec.errors import ClusterError
from warprec.utils.enums import ErroredTrialPolicy
from warprec.utils.pause import is_pause_requested
from warprec.utils.helpers import validation_metric
from warprec.utils.callback import WarpRecCallback
from warprec.utils.logger import logger
from warprec.utils.registry import (
    model_registry,
    search_algorithm_registry,
    scheduler_registry,
    search_space_registry,
)

# Optional imports handling
try:
    from warprec.recommenders.trainer.dashboard_callbacks import (
        CodeCarbonCallback,
        WarpRecMLflowLoggerCallback,
        WarpRecWandbLoggerCallback,
    )
    from warprec.recommenders.trainer.emissions import EmissionsSettings

    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


def _label_matches(rule: str, value: Optional[str]) -> bool:
    """Whether a node label satisfies one rule of a Ray label selector.

    Args:
        rule (str): The rule: 'value', '!value', 'in(a,b)' or '!in(a,b)'.
        value (Optional[str]): The node's value for that label, None when unset.

    Returns:
        bool: True when the node satisfies the rule.
    """
    negated = rule.startswith("!")
    body = rule[1:] if negated else rule
    if body.startswith("in(") and body.endswith(")"):
        allowed = {item.strip() for item in body[3:-1].split(",")}
    else:
        allowed = {body}
    return (value in allowed) != negated


def trials_that_fit(
    nodes: List[Dict[str, Any]],
    scaling_config_dict: Dict[str, Any],
    trainer_driver_cpu: float = 0.05,
) -> int:
    """How many trials the nodes of a cluster can run at once.

    A trial is a group of workers, each of which has to fit on one node. Only
    the live nodes whose labels match the trial's label selector and that offer
    every resource a worker asks for count; each of them hosts as many workers
    as its scarcest resource allows. The tiny CPU of every trial's Tune driver
    can go to any node.

    Args:
        nodes (List[Dict[str, Any]]): The nodes, as 'ray.nodes()' gives them.
        scaling_config_dict (Dict[str, Any]): The resources of a trial.
        trainer_driver_cpu (float): The CPU of the trial's Tune driver.

    Returns:
        int: The number of trials.

    Raises:
        ClusterError: If no node can host a worker of the trial.
    """
    per_worker = {
        name: amount
        for name, amount in scaling_config_dict["resources_per_worker"].items()
        if amount > 0
    }
    num_workers = scaling_config_dict["num_workers"]
    selector = scaling_config_dict.get("label_selector") or {}

    live = [node for node in nodes if node.get("Alive", True)]
    workers = 0
    for node in live:
        labels = node.get("Labels") or {}
        if not all(
            _label_matches(rule, labels.get(key)) for key, rule in selector.items()
        ):
            continue
        resources = node.get("Resources") or {}
        # A small tolerance, so that 1.0 / 0.2 does not count four workers
        workers += min(
            (
                int(resources.get(name, 0.0) / amount + 1e-9)
                for name, amount in per_worker.items()
            ),
            default=0,
        )

    if workers < num_workers:
        where = f" on the nodes labelled {selector}" if selector else ""
        raise ClusterError(
            f"No trial can run: each needs {num_workers} worker(s) of "
            f"{per_worker}{where}, and the cluster offers room for {workers}. "
            "Lower the resources per trial or add nodes that have them."
        )

    trials = workers // num_workers
    total_cpu = sum((node.get("Resources") or {}).get("CPU", 0.0) for node in live)
    trial_cpu = per_worker.get("CPU", 0.0) * num_workers + trainer_driver_cpu
    if trial_cpu > 0:
        trials = min(trials, int(total_cpu / trial_cpu + 1e-9))
    return max(trials, 1)


@dataclass(frozen=True)
class TrainingOutcome:
    """The result of a hyperparameter optimization run for one model.

    Attributes:
        status (str): One of 'completed', 'interrupted' or 'failed'.
        model (Optional[Recommender]): The best model, when one was produced.
        best_params (Optional[dict]): The best hyperparameters found.
        report (dict): Summary report of the optimization.
        best_iter (int): The best training iteration.
        best_checkpoint_path (Optional[str]): Path of the best trial checkpoint.
    """

    status: str
    model: Optional[Recommender] = None
    best_params: Optional[dict] = None
    report: dict = field(default_factory=dict)
    best_iter: int = 0
    best_checkpoint_path: Optional[str] = None

    @property
    def interrupted(self) -> bool:
        """Returns whether the optimization was interrupted by a pause.

        Returns:
            bool: True when the run was paused before completing.
        """
        return self.status == "interrupted"

    @property
    def failed(self) -> bool:
        """Returns whether the optimization produced no usable model.

        Returns:
            bool: True when the optimization failed.
        """
        return self.status == "failed"


class Trainer:
    """Trainer class for training and hyperparameter optimization using Ray Tune.
    Delegates configuration details to TrainConfiguration object.

    Args:
        storage_path (str): Path to store Ray Tune results and checkpoints.
        custom_callback (WarpRecCallback): Custom callback for training/eval.
        custom_modules (Optional[Union[str, List[str]]]): List of custom models to load.
        dashboard_config (Optional[DashboardConfig]): Configuration for logging dashboards
            (WandB, MLFlow, CodeCarbon). Defaults to None.
        run_name (Optional[str]): The identifier of the run. When provided, Ray Tune
            experiments are given deterministic names so that an interrupted run
            can be restored. When None, Ray Tune generates a random experiment
            name and the run cannot be resumed.
        errored_trials (ErroredTrialPolicy): How to treat trials that errored
            before a pause when restoring a run. Defaults to skipping them.
        resumable (bool): Whether the run state records this model's Ray Tune
            experiment as one to continue. When False, an experiment left
            under the same name by an earlier run is moved aside rather than
            restored, so that it cannot leak an old split or configuration
            into this one. Defaults to False.
        emissions (Optional[EmissionsSettings]): The CodeCarbon measurement of
            the run the trainer belongs to, when the run measures itself (the
            train and swarm pipelines do). Without one, a search with CodeCarbon
            enabled measures itself. Defaults to None.
        gpu_deterministic (bool): Whether the trials use deterministic
            algorithms only, as 'general.gpu_deterministic' sets it. Defaults
            to False.
        concurrent_searches (int): How many searches share the cluster at once,
            as in the swarm. Each gets an equal share of it when the number of
            concurrent trials is estimated. Defaults to 1.
    """

    def __init__(
        self,
        storage_path: str,
        custom_callback: WarpRecCallback = WarpRecCallback(),
        custom_modules: Optional[Union[str, List[str]]] = None,
        dashboard_config: Optional[DashboardConfig] = None,
        run_name: Optional[str] = None,
        errored_trials: ErroredTrialPolicy = ErroredTrialPolicy.SKIP,
        resumable: bool = False,
        emissions: Optional["EmissionsSettings"] = None,
        gpu_deterministic: bool = False,
        concurrent_searches: int = 1,
    ):
        self._stg_path = storage_path
        self._concurrent_searches = max(1, concurrent_searches)
        self._gpu_deterministic = gpu_deterministic
        self._custom_modules = custom_modules or []
        self._run_name = run_name
        self._callbacks = self._setup_callbacks(
            custom_callback, dashboard_config, emissions
        )
        self._errored_trials = errored_trials
        self._resumable = resumable
        self._spans = TrialSpans()

    def experiment_name(self, model_name: str) -> Optional[str]:
        """Builds the deterministic Ray Tune experiment name for a model.

        Args:
            model_name (str): The name of the model.

        Returns:
            Optional[str]: The experiment name, or None when the trainer has no
                run name and Ray Tune should generate one.
        """
        if self._run_name is None:
            return None
        return f"{self._run_name}__{model_name}"

    @staticmethod
    def is_interrupted(results: tune.ResultGrid) -> bool:
        """Detects whether a Ray Tune sweep was interrupted before finishing.

        Ray Tune returns from 'fit' normally after an interrupted sweep rather
        than raising, and the trials it reports afterwards have already been
        cleaned up, so their states do not distinguish a paused sweep from a
        finished one. The pause controller is therefore the authoritative
        signal; the trial states are only a fallback for the case where the
        sweep was stopped without one installed.

        Args:
            results (tune.ResultGrid): The result grid returned by 'Tuner.fit'.

        Returns:
            bool: True when the sweep was interrupted.
        """
        if is_pause_requested():
            return True

        total = len(results)
        if total == 0:
            return False
        return (results.num_terminated + results.num_errors) < total

    def train_single_fold(
        self,
        model_name: str,
        params: RecomModel,
        dataset: Union[Dataset, ray.ObjectRef],
        evaluation: EvaluationConfig,
        validation_score: str,
        device: str = "cpu",
        ray_verbose: int = 1,
    ) -> TrainingOutcome:
        """Main method of the Trainer class.

        This method will execute the training of the model and evaluation,
        according to information passed through configuration.

        Args:
            model_name (str): The name of the model to optimize.
            params (RecomModel): The parameters of the model.
            dataset (Union[Dataset, ray.ObjectRef]): The dataset to use during
                training, or a reference to it in the object store, which the
                trials read without it being stored again.
            evaluation (EvaluationConfig): The evaluation section of the configuration.
            validation_score (str): The metric to monitor during training.
            device (str): The device that will be used for tensor operations.
            ray_verbose (int): The Ray level of verbosity.

        Returns:
            TrainingOutcome: The outcome of the optimization, carrying the best
                parameters, the summary report, the best training iteration and
                the checkpoint to build the best model from, but not the model,
                which 'load_best_model' builds. When the sweep was paused or
                produced no usable result, the outcome carries the corresponding
                status.
        """

        mode = params.optimization.properties.mode

        tuner = self._setup_tuner(
            model_name=model_name,
            params=params,
            dataset=dataset,
            evaluation=evaluation,
            validation_score=validation_score,
            device=device,
            ray_verbose=ray_verbose,
        )

        search_start = time.perf_counter()
        results = tuner.fit()
        search_s = time.perf_counter() - search_start

        # Ray Tune returns normally from an interrupted sweep, so a partial
        # result grid must never be mistaken for a finished optimization.
        if self.is_interrupted(results):
            logger.attention(
                f"Hyperparameter optimization for {model_name} was interrupted. "
                "No best model will be selected from this partial sweep."
            )
            return TrainingOutcome(status="interrupted")

        # A trial that fails reports a score of -inf (+inf when minimizing)
        # rather than raising, so Ray Tune counts no error and would pick it
        best = self._best_usable_trial(
            results,
            validation_score,
            mode,
            needs_checkpoint=issubclass(
                model_registry.get_class(model_name), IterativeRecommender
            ),
        )
        if best is None:
            logger.negative(
                f"All trials failed for {model_name}. "
                f"First error: {self._first_trial_error(results)}"
            )
            return TrainingOutcome(status="failed")

        best_result, best_row = best
        best_params = best_result.config
        best_params = {k: v for k, v in best_params.items() if not k.startswith("_")}
        best_score = best_row.get(validation_score).item()
        best_iter = self._training_iterations(best_row)
        best_checkpoint_path = best_row.get("checkpoint_path")

        logger.msg(
            f"Best params: {best_params} | Score ({validation_score}): {best_score} "
            f"| Iteration: {best_iter}"
        )
        logger.positive(f"HPO for {model_name} ended successfully.")

        # The best model is built by the caller once it has recorded the search,
        # so that a failure to build it does not cost the search
        report = self._create_report(
            results, [best_result.metrics["trial_id"]], search_s
        )
        return TrainingOutcome(
            status="completed",
            best_params=best_params,
            report=report,
            best_iter=best_iter,
            best_checkpoint_path=str(best_checkpoint_path),
        )

    def train_multiple_fold(
        self,
        model_name: str,
        params: RecomModel,
        datasets: Union[List[Dataset], ray.ObjectRef],
        evaluation: EvaluationConfig,
        validation_score: str,
        device: str = "cpu",
        desired_training_it: str = "median",
        ray_verbose: int = 1,
        num_folds: Optional[int] = None,
    ) -> TrainingOutcome:
        """Main method of the Trainer class for cross-validation.

        Args:
            model_name (str): The name of the model to optimize.
            params (RecomModel): The parameters of the model.
            datasets (Union[List[Dataset], ray.ObjectRef]): The folds to use during
                training, or a reference to their list in the object store.
            evaluation (EvaluationConfig): The evaluation section of the configuration.
            validation_score (str): The metric to monitor during training.
            device (str): The device that will be used for tensor operations.
            desired_training_it (str): The type of statistic to use to
                select the number of training iterations to use
                when training on the full dataset. Either "min", "max",
                "mean" or "median". Default is "median".
            ray_verbose (int): The Ray level of verbosity.
            num_folds (Optional[int]): The number of folds, required when the
                folds are given by reference.

        Returns:
            TrainingOutcome: The outcome of the optimization, carrying the best
                hyperparameters and the summary report. Cross-validation
                produces no single best checkpoint, so the outcome carries no
                model. When the sweep was paused or produced no usable result,
                the outcome carries the corresponding status.

        Raises:
            ValueError: If the folds are given by reference without their number.
        """
        if not isinstance(datasets, list) and not num_folds:
            raise ValueError("The number of folds is needed with a reference to them.")

        mode = params.optimization.properties.mode

        tuner = self._setup_tuner(
            model_name=model_name,
            params=params,
            dataset=datasets,
            evaluation=evaluation,
            validation_score=validation_score,
            device=device,
            ray_verbose=ray_verbose,
            num_folds=len(datasets) if isinstance(datasets, list) else num_folds,
        )

        search_start = time.perf_counter()
        results = tuner.fit()
        search_s = time.perf_counter() - search_start

        # Ray Tune returns normally from an interrupted sweep, so a partial
        # result grid must never be mistaken for a finished optimization.
        if self.is_interrupted(results):
            logger.attention(
                f"Cross-validated optimization for {model_name} was interrupted. "
                "No best hyperparameters will be selected from this partial sweep."
            )
            return TrainingOutcome(status="interrupted")

        result_df = results.get_dataframe(
            filter_metric=validation_score, filter_mode=mode
        )

        # A failed trial reports an infinite score, whichever the mode
        if (
            result_df.empty
            or not np.isfinite(
                pd.to_numeric(result_df[validation_score], errors="coerce")
            ).any()
        ):
            logger.negative(
                f"All trials failed for {model_name}. "
                f"First error: {self._first_trial_error(results)}"
            )
            return TrainingOutcome(status="failed")

        best_rows = {
            result.metrics["trial_id"]: self._get_best_result_row(
                self._trial_history(result), validation_score, mode
            )
            for result in results  # type: ignore[attr-defined]
            if result.metrics
        }
        trial_rows = [best_rows[trial_id] for trial_id in result_df["trial_id"]]
        result_df[validation_score] = [row[validation_score] for row in trial_rows]
        result_df["iterations"] = [self._training_iterations(row) for row in trial_rows]

        # Aggregate results logic
        best_hyperparameters, best_stats = self._aggregate_cv_results(
            result_df, validation_score, mode, desired_training_it
        )

        hyperparameters = {
            k: v for k, v in best_hyperparameters.items() if k != "iterations"
        }
        logger.msg(
            f"Best params: {hyperparameters} | Avg Score: {best_stats['mean']} "
            f"| Std: {best_stats['std']} | Iterations: {best_hyperparameters['iterations']}"
        )
        logger.positive(f"CV HPO for {model_name} ended successfully.")

        report = self._create_report(results, best_stats["trial_ids"], search_s)
        return TrainingOutcome(
            status="completed",
            best_params=best_hyperparameters,
            report=report,
            best_iter=best_hyperparameters["iterations"],
        )

    def _setup_tuner(
        self,
        model_name: str,
        params: RecomModel,
        dataset: Union[Dataset, List[Dataset], ray.ObjectRef],
        evaluation: EvaluationConfig,
        validation_score: str,
        device: str,
        ray_verbose: int,
        num_folds: Optional[int] = None,
    ) -> Tuner:
        """Prepares the Ray Tuner instance.

        Args:
            model_name (str): The name of the model to optimize.
            params (RecomModel): The parameters of the model.
            dataset (Union[Dataset, List[Dataset], ray.ObjectRef]): The dataset(s)
                to use during training, or a reference to them in the object store.
            evaluation (EvaluationConfig): The evaluation section of the configuration.
            validation_score (str): The metric to monitor during training.
            device (str): The device that will be used for tensor operations.
            ray_verbose (int): The Ray level of verbosity.
            num_folds (Optional[int]): The number of cross-validation folds, None
                or 0 for a single split.

        Returns:
            Tuner: The configured Ray Tuner instance.
        """

        opt_config = params.optimization
        mode = opt_config.properties.mode

        # Determine resources
        scaling_config_dict = self._get_scaling_config(
            opt_config.cpu_per_trial,
            opt_config.gpu_per_trial,
            opt_config.custom_resources_per_trial,
            opt_config.label_selector,
            device,
        )

        # Prepare data bundle
        validation_metric_name, validation_top_k = validation_metric(validation_score)
        data_bundle = {
            "model_name": model_name,
            # A dataset already in the object store is not stored again
            "dataset_folds": (
                dataset if isinstance(dataset, ray.ObjectRef) else ray.put(dataset)
            ),
            "evaluation": evaluation,
            "validation_top_k": validation_top_k,
            "validation_metric_name": validation_metric_name,
            "mode": opt_config.properties.mode,
            "device": device,
            "eval_every_n": opt_config.eval_every_n,
            "lr_scheduler": opt_config.lr_scheduler,
            "optimizer": opt_config.optimizer,
            "seed": opt_config.properties.seed,
            "block_size": opt_config.block_size,
            "chunk_size": opt_config.chunk_size,
            "custom_modules": self._custom_modules,
            "gpu_deterministic": self._gpu_deterministic,
            "early_stopping_config": params.early_stopping,
            # The trial rebuilds its configuration from the sampled
            # hyperparameters alone, so precision and clipping travel here
            "optimization": opt_config,
            "num_workers": opt_config.num_workers,
            "cpu_per_worker": (
                scaling_config_dict.get("resources_per_worker") or {}
            ).get("CPU"),
            # CodeCarbon's 'process' tracking has to run where the trial trains
            "codecarbon": next(
                (
                    callback.trial_settings(scaling_config_dict["num_workers"])
                    for callback in self._callbacks
                    if DASHBOARD_AVAILABLE and isinstance(callback, CodeCarbonCallback)
                ),
                None,
            ),
        }

        if isinstance(dataset, list):
            num_folds = len(dataset)
        param_space = self._parse_params(params, num_folds or 0)
        experiment_name = self.experiment_name(model_name)
        restore = self._claim_experiment(experiment_name)
        logged_epochs = (
            read_logged_epochs(os.path.join(self._stg_path, experiment_name))
            if restore and experiment_name
            else {}
        )
        session_id = uuid.uuid4().hex

        # The trial function is pickled into every trial, so it must not
        # reference 'self': that would ship the whole Trainer, the custom and
        # dashboard callbacks included, to processes that never call them
        storage_path = self._stg_path

        # Driver function that will be launched by Ray Tune and will
        # initialize Ray Train
        def train_driver_fn(config: dict, data_bundle: dict, scaling_config_dict: dict):
            scaling_config = ScalingConfig(**scaling_config_dict)
            trial_id = tune.get_context().get_trial_id()

            train_loop_config = {
                **data_bundle,
                "params": config,
                "last_logged_epoch": logged_epochs.get(trial_id),
                "session_id": session_id,
                "trial_id": trial_id,
                # The trial's setup is counted from here, on the worker
                TRIAL_STARTED_AT: time.time(),
            }

            trainer = TorchTrainer(
                train_loop_per_worker=objective_function,
                train_loop_config=train_loop_config,
                scaling_config=scaling_config,
                run_config=ray.train.RunConfig(
                    name=f"ray_train_trial_id={trial_id}",
                    callbacks=[TuneReportCallback()],  # type: ignore[list-item]
                    storage_path=storage_path,
                    checkpoint_config=ray.train.CheckpointConfig(
                        num_to_keep=opt_config.checkpoint_to_keep,
                        checkpoint_score_attribute=validation_score,
                        checkpoint_score_order=mode,
                    ),
                ),
            )

            trainer.fit()

        # Search Algorithm & Scheduler
        # Estimated even when the limit is set, so that a trial no node can
        # host stops the run here rather than waiting forever to be scheduled
        estimated = self._estimate_max_concurrent_trials(scaling_config_dict)
        max_concurrent_trials = (
            opt_config.max_concurrent_trials
            if opt_config.max_concurrent_trials is not None
            else estimated
        )
        search_alg: Any = search_algorithm_registry.get(
            opt_config.strategy,
            metric=validation_score,
            max_concurrent_trials=max_concurrent_trials,
            **opt_config.properties.model_dump(),
        )
        # A model whose hyperparameters constrain each other has the
        # combinations it refuses skipped before a trial is launched for them
        if params.need_single_trial_validation:
            if isinstance(search_alg, CappedVariantGenerator):
                search_alg.model_name = model_name
            else:
                search_alg = ValidSuggestions(search_alg, model_name)
        # Capped by a limiter that a restored search does not reset to its old
        # cap, so that a search resumed on another cluster runs as many trials
        # as that cluster can hold
        if max_concurrent_trials and not isinstance(search_alg, BasicVariantGenerator):
            search_alg = ConcurrencyCap(
                search_alg, max_concurrent=max_concurrent_trials
            )
        scheduler = scheduler_registry.get(
            opt_config.scheduler, **opt_config.properties.model_dump()
        )

        # Configs. A fresh record of when the trials run, for this search only
        self._spans = TrialSpans()
        run_config = ray.tune.RunConfig(
            name=self.experiment_name(model_name),
            callbacks=[*self._callbacks, self._spans],
            verbose=ray_verbose,
            storage_path=self._stg_path,
            checkpoint_config=ray.tune.CheckpointConfig(
                num_to_keep=opt_config.checkpoint_to_keep,
                checkpoint_score_attribute=validation_score,
                checkpoint_score_order=mode,
            ),
        )

        # Tuner will execute the driver function
        tune_config = TuneConfig(
            metric=validation_score,
            mode=mode,
            search_alg=search_alg,  # type: ignore[arg-type]
            scheduler=scheduler,  # type: ignore[arg-type]
            num_samples=opt_config.num_samples,
            # The searcher caps itself, and Ray logs that it is ignoring the
            # value when it gets one it cannot apply.
            max_concurrent_trials=None,
            trial_name_creator=self._trial_name_creator(model_name),
            trial_dirname_creator=self._trial_dirname_creator(model_name),
        )
        trainable = tune.with_resources(
            tune.with_parameters(
                train_driver_fn,
                data_bundle=data_bundle,
                scaling_config_dict=scaling_config_dict,
            ),
            # Reserve only the lightweight Tune driver resources here.
            # The inner TorchTrainer will request the actual worker bundles.
            resources={"CPU": 0.05},
        )
        # Ray Tune registers a trainable in the cluster under its name, once
        # per job: searches that run at once (the swarm's) need names of their
        # own, or each overwrites the others and its trials run another model
        stored_name = (
            self._stored_trainable_name(os.path.join(self._stg_path, experiment_name))
            if restore and experiment_name
            else None
        )
        trainable.__name__ = stored_name or self._trainable_name(
            model_name, experiment_name, session_id
        )

        return self._build_or_restore_tuner(
            trainable=trainable,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
            experiment_name=experiment_name,
            restore=restore,
            max_concurrent_trials=max_concurrent_trials,
        )

    @staticmethod
    def _trainable_name(
        model_name: str, experiment_name: Optional[str], session_id: str
    ) -> str:
        """The name a search's trainable is registered under with Ray Tune.

        It is unique to the model's search in its run, so that searches running
        at once do not share one, and the same every time the run is resumed,
        since a restored search looks its trainable up by the name it was
        created with.

        Args:
            model_name (str): The name of the model.
            experiment_name (Optional[str]): The deterministic experiment name,
                or None when the run is not resumable.
            session_id (str): The id of this session, which names the search
                when it has no experiment name.

        Returns:
            str: The name, in the ASCII characters Ray Tune's registry accepts.
        """
        identity = experiment_name or f"{model_name}__{session_id}"
        readable = re.sub(r"[^A-Za-z0-9_.-]", "_", identity)
        digest = hashlib.sha1(
            identity.encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:8]
        return f"warprec__{readable}__{digest}"

    @staticmethod
    def _stored_trainable_name(experiment_path: str) -> Optional[str]:
        """The name the trainable of a stored search was registered under.

        A search is restored only with a trainable of the same name; one
        paused by WarpRec 2.2.2 or earlier used 'train_driver_fn'.

        Args:
            experiment_path (str): The path of the Ray Tune experiment.

        Returns:
            Optional[str]: The name, None when the experiment does not say.
        """
        fs, fs_path = get_fs_and_path(experiment_path)
        try:
            with fs.open_input_file(f"{fs_path}/tuner.pkl") as stored:
                state = cloudpickle.loads(stored.readall())
        except (OSError, EOFError, cloudpickle.pickle.UnpicklingError):
            return None
        name = state.get("__trainable_name") if isinstance(state, dict) else None
        return name if isinstance(name, str) else None

    def _claim_experiment(self, experiment_name: Optional[str]) -> bool:
        """Decides whether to restore the Ray Tune experiment of a model.

        Only an experiment the run state records as resumable is restored. Any
        other experiment found under the same name was left by a run whose
        state has been discarded, so it is moved aside to
        '<name>.discarded-<timestamp>' rather than deleted, and the model
        starts afresh.

        Args:
            experiment_name (Optional[str]): The deterministic experiment name,
                or None when the run is not resumable.

        Returns:
            bool: True when the existing experiment should be restored.
        """
        if experiment_name is None:
            return False

        experiment_path = os.path.join(self._stg_path, experiment_name)
        if self._resumable and Tuner.can_restore(experiment_path):
            return True

        # Resolved the way Ray Tune resolves it, so remote storage works too
        fs, fs_path = get_fs_and_path(experiment_path)
        if fs.get_file_info(fs_path).type == pyarrow.fs.FileType.NotFound:
            return False

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        aside, suffix = f"{fs_path}.discarded-{stamp}", 1
        while fs.get_file_info(aside).type != pyarrow.fs.FileType.NotFound:
            aside = f"{fs_path}.discarded-{stamp}-{suffix}"
            suffix += 1
        fs.move(fs_path, aside)
        logger.attention(
            f"The Ray Tune experiment at {experiment_path} belongs to run state "
            f"that was discarded. It was moved to {aside} and the model will be "
            "optimized from scratch."
        )
        return False

    def _build_or_restore_tuner(
        self,
        trainable: Any,
        param_space: dict,
        tune_config: TuneConfig,
        run_config: Any,
        experiment_name: Optional[str],
        restore: bool = False,
        max_concurrent_trials: Optional[int] = None,
    ) -> Tuner:
        """Builds a new Tuner, or restores the one left behind by a paused run.

        The trainable is rebuilt rather than reused, because 'tune.with_parameters'
        captures a fresh object store reference for the dataset in every process.

        Args:
            trainable (Any): The trainable to run.
            param_space (dict): The search space of the optimization.
            tune_config (TuneConfig): The tuning configuration.
            run_config (Any): The Ray Tune run configuration.
            experiment_name (Optional[str]): The deterministic experiment name,
                or None when the run is not resumable.
            restore (bool): Whether to restore the experiment, as decided by
                '_claim_experiment'.
            max_concurrent_trials (Optional[int]): The concurrency cap computed
                for this session, which a restored search takes over in place of
                the one it was paused with.

        Returns:
            Tuner: A fresh Tuner, or one restored from a previous run.
        """
        if restore and experiment_name is not None:
            experiment_path = os.path.join(self._stg_path, experiment_name)
            logger.positive(
                f"Found a resumable Ray Tune experiment at {experiment_path}. "
                "Unfinished trials will continue from their last checkpoint."
            )
            tuner = Tuner.restore(
                experiment_path,
                trainable=trainable,
                param_space=param_space,
                resume_unfinished=True,
                resume_errored=self._errored_trials == ErroredTrialPolicy.RESUME,
                restart_errored=self._errored_trials == ErroredTrialPolicy.RESTART,
            )
            if max_concurrent_trials is not None:
                recap_restored_search(tuner, max_concurrent_trials)
            return tuner

        return Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )

    def _get_scaling_config(
        self,
        cpu_per_trial: int,
        gpu_per_trial: float,
        custom_resources_per_trial: Optional[Dict[str, float | int]] = None,
        label_selector: Optional[Dict[str, str]] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Calculates resource allocation per trial.

        Args:
            cpu_per_trial (int): The number of cpu per trial.
            gpu_per_trial (float): The number of gpu per trial.
            custom_resources_per_trial (Optional[Dict[str, float | int]]): Custom resources to assign to each trial.
            label_selector (Optional[Dict[str, str]]): Custom labels used during trial assignment.
            device (str): The device of the experiment.

        Returns:
            Dict[str, Any]: Resources dictionary for Ray Tune.
        """
        # Determine if we are using GPUs
        use_gpu = device == "cuda" or gpu_per_trial > 0

        # In Ray Train, `num_workers` represents the number of distributed processes.
        # For GPU training, we typically spawn 1 worker per GPU to enable DDP.
        if use_gpu:
            num_workers = max(1, int(gpu_per_trial))
        else:
            # For CPU training, we default to 1 worker per trial
            num_workers = 1

        # Calculate resources per individual worker
        resources_per_worker = {
            "CPU": cpu_per_trial / num_workers,
        }
        if use_gpu:
            resources_per_worker["GPU"] = gpu_per_trial / num_workers

        # Add custom resources if provided
        if custom_resources_per_trial is not None:
            for res, amount in custom_resources_per_trial.items():
                resources_per_worker[res] = amount / num_workers

        scaling_dict = {
            "num_workers": num_workers,
            "use_gpu": use_gpu,
            "resources_per_worker": resources_per_worker,
        }

        # Add label selector if provided
        if label_selector is not None:
            scaling_dict["label_selector"] = label_selector

        return scaling_dict

    def _estimate_max_concurrent_trials(
        self,
        scaling_config_dict: Dict[str, Any],
        trainer_driver_cpu: float = 0.05,
    ) -> int:
        """Estimate a safe Tune concurrency limit from Ray Train trial resources.

        The Tune trial itself only reserves a tiny driver CPU. The real cluster
        footprint comes from the inner TorchTrainer worker group, so concurrency
        must be capped explicitly to avoid launching more Train runs than the
        cluster can actually execute in parallel. The cluster is read as it is
        when the search starts, or is restored. A search running alongside
        others, as in the swarm, gets an equal share of it.

        Args:
            scaling_config_dict (Dict[str, Any]): The resources of a trial.
            trainer_driver_cpu (float): The CPU of the trial's Tune driver.

        Returns:
            int: How many trials can run at once, at least 1.
        """
        trials = trials_that_fit(ray.nodes(), scaling_config_dict, trainer_driver_cpu)
        return max(1, trials // self._concurrent_searches)

    @staticmethod
    def load_best_model(
        model_name: str,
        checkpoint_path: str,
        best_params: dict,
        dataset: Dataset,
    ) -> Recommender:
        """Loads the model state from a trial checkpoint.

        A closed-form model has no checkpoint: it is refitted from its parameters.
        The pipelines call this on the cluster, where the dataset is.

        Args:
            model_name (str): The name of the model.
            checkpoint_path (str): The path of the checkpoint to load.
            best_params (dict): The hyperparameters of the model.
            dataset (Dataset): The dataset the model was trained on.

        Returns:
            Recommender: The reconstructed model.
        """
        model_class = model_registry.get_class(model_name)

        if issubclass(model_class, IterativeRecommender):
            model = model_class.from_checkpoint(
                torch.load(
                    Path(checkpoint_path) / "checkpoint.ckpt",
                    weights_only=False,
                    map_location="cpu",
                ),
                interactions=dataset.train_set,
                transactions=dataset.train_transactions,
                knowledge=dataset.knowledge,
                multimodal=dataset.multimodal,
                sessions=dataset.train_session,
                **check_stash(dataset.get_stash(), MODEL_ARGUMENTS, "model"),
            )
        else:
            model = model_class(
                params=best_params,
                interactions=dataset.train_set,
                transactions=dataset.train_transactions,
                knowledge=dataset.knowledge,
                multimodal=dataset.multimodal,
                sessions=dataset.train_session,
                info=dataset.info(),
                **check_stash(dataset.get_stash(), MODEL_ARGUMENTS, "model"),
            )

        return model

    def _get_best_result_row(self, df, metric, mode):
        """Extract the best iteration row from the result DataFrame."""
        if mode == "max":
            best_idx = df[metric].idxmax()
        else:
            best_idx = df[metric].idxmin()
        return df.loc[best_idx]

    def _best_usable_trial(
        self,
        results: tune.ResultGrid,
        metric: str,
        mode: str,
        needs_checkpoint: bool,
    ) -> Optional[tuple]:
        """The best report of the search that a best model can be built from.

        A report counts when its score is finite and, for an iterative model,
        it came with a checkpoint: a failed trial reports an infinite score
        without one, and a diverged one a NaN.

        Args:
            results (tune.ResultGrid): The results of the search.
            metric (str): The validation metric.
            mode (str): 'max' or 'min'.
            needs_checkpoint (bool): Whether the model is rebuilt from a checkpoint.

        Returns:
            Optional[tuple]: The best trial's result and its best report, or None
                when no trial produced a usable one.
        """
        best: Optional[tuple] = None
        for result in results:  # type: ignore[attr-defined]
            if not result.metrics:
                continue
            history = self._trial_history(result)
            if metric not in history:
                continue
            scores = pd.to_numeric(history[metric], errors="coerce")
            usable = history[np.isfinite(scores)]
            if needs_checkpoint:
                if "checkpoint_path" not in usable:
                    continue
                usable = usable[usable["checkpoint_path"].notna()]
            if usable.empty:
                continue

            row = self._get_best_result_row(usable, metric, mode)
            score = float(row[metric])
            if best is None or (score > best[2] if mode == "max" else score < best[2]):
                best = (result, row, score)
        return best[:2] if best is not None else None

    @staticmethod
    def _first_trial_error(results: tune.ResultGrid) -> str:
        """What made the first failed trial of a search fail.

        Args:
            results (tune.ResultGrid): The results of the search.

        Returns:
            str: The error, as the trial reported it or as Ray Tune caught it.
        """
        for result in results:  # type: ignore[attr-defined]
            if result.metrics and result.metrics.get(TRIAL_ERROR):
                return str(result.metrics[TRIAL_ERROR])
            if result.error:
                return str(result.error)
        return "no trial reported a finite score"

    def _trial_history(self, result):
        """Returns the reports of a trial, including those made before a pause."""
        history = result.metrics_dataframe
        if "checkpoint_path" not in history:
            return history

        logging.getLogger(
            "ray.train.v2._internal.execution.checkpoint.checkpoint_manager"
        ).setLevel(logging.WARNING)
        run = ray.train.Result.from_path(
            os.path.join(
                self._stg_path, f"ray_train_trial_id={result.metrics['trial_id']}"
            )
        )
        history = run.metrics_dataframe
        history["checkpoint_path"] = [
            checkpoint.path for checkpoint, _ in run.best_checkpoints
        ]
        return history

    @staticmethod
    def _training_iterations(row) -> int:
        """Counts the epochs trained up to a report, which survive a trial resume."""
        if "epoch" in row and not math.isnan(row["epoch"]):
            return int(row["epoch"]) + 1
        return int(row["training_iteration"])

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

        aggregations = {
            "mean_score": (metric, "mean"),
            "std_score": (metric, "std"),
            "desired_training_iterations": ("iterations", desired_it_stat),
        }
        if "trial_id" in df:
            aggregations["trial_ids"] = ("trial_id", list)
        agg_df = (
            df.groupby(hyperparam_cols or (lambda _: 0))
            .agg(**aggregations)
            .reset_index()
        )

        best_row = agg_df.sort_values(by="mean_score", ascending=mode == "min").iloc[0]

        # Reconstruct clean params dict
        best_params = {"iterations": math.ceil(best_row["desired_training_iterations"])}
        for col in hyperparam_cols:
            key = col.replace("config/", "")
            val = best_row[col]
            # Type restoration logic. The row mixes the hyperparameters, so
            # pandas may have widened an integer to a float there: the type is
            # read from the hyperparameter's own column instead
            kind = df[col].dtype.kind
            if kind == "b":
                best_params[key] = bool(val)
            elif kind in "iu":
                best_params[key] = int(val)
            elif kind == "f":
                # A float column is a float hyperparameter, unless it holds the
                # integers of a hyperparameter some trials lack (pandas turns
                # an integer column with missing values into floats)
                was_int = df[col].isna().any() and all(
                    float(v).is_integer() for v in df[col].dropna()
                )
                best_params[key] = int(val) if was_int else float(val)
            else:
                best_params[key] = val

        stats = {
            "mean": best_row["mean_score"],
            "std": best_row["std_score"],
            # The folds of the best configuration, its trials
            "trial_ids": list(best_row.get("trial_ids", [])),
        }
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
                tune_params[k] = search_space_registry.get(space_type)(*args)
            else:
                tune_params[k] = v  # Static parameter

        if num_folds > 0:
            tune_params["fold"] = tune.grid_search(list(range(num_folds)))

        return tune_params

    def _trial_name_creator(self, model_name: str):
        def _creator(trial: Trial):
            model_class = model_registry.get_class(model_name)
            return model_class.get_name_from_params(trial.config)

        return _creator

    def _trial_dirname_creator(self, model_name: str):
        def _creator(trial: Trial):
            model_class = model_registry.get_class(model_name)
            name_ext = model_class.get_name_from_params(trial.config)
            return f"{name_ext}_{trial.trial_id}"

        return _creator

    def _setup_callbacks(
        self,
        custom_callback: WarpRecCallback,
        dashboard: Optional[DashboardConfig] = None,
        emissions: Optional["EmissionsSettings"] = None,
    ) -> List[tune.Callback | WarpRecCallback]:
        callbacks: List[tune.Callback | WarpRecCallback] = [
            custom_callback,
            WarpRecJsonLoggerCallback(),
            WarpRecCSVLoggerCallback(),
        ]

        # If dashboard config is not provided, return only the custom callback
        if dashboard is None:
            return callbacks

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
                WarpRecWandbLoggerCallback(
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
        # Under 'machine' tracking a run that measures itself already covers
        # its searches: there is nothing left for a search to measure
        if dashboard.codecarbon.enabled and not (
            emissions is not None and emissions.tracking_mode == "machine"
        ):
            callbacks.append(
                CodeCarbonCallback(
                    save_to_api=dashboard.codecarbon.save_to_api,
                    save_to_file=dashboard.codecarbon.save_to_file,
                    output_dir=self._stg_path,
                    tracking_mode=dashboard.codecarbon.tracking_mode,
                    project_name=self._run_name,
                    settings=emissions,
                )
            )
        if dashboard.mlflow.enabled:
            callbacks.append(
                WarpRecMLflowLoggerCallback(
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
        best_trial_ids: List[str],
        search_s: float,
        model: Optional[Recommender] = None,
    ) -> dict:
        """Summarises the memory, time and size of a finished search.

        Args:
            results (tune.ResultGrid): The results of the search.
            best_trial_ids (List[str]): The best trial, or under cross-validation
                the trials of the best configuration, one per fold.
            search_s (float): The wall clock of the search, from 'Tuner.fit'.
            model (Optional[Recommender]): The best model, when there is one.

        Returns:
            dict: The report, by time report column.
        """
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

        # Where the trials' time went, from what every epoch reported
        timing = {
            r.metrics["trial_id"]: trial_timing(r.metrics_dataframe)
            for r in results  # type: ignore[attr-defined]
            if r.metrics and "trial_id" in r.metrics
        }
        report.update(summarise_timing(timing, best_trial_ids))
        report[SEARCH_OVERHEAD] = search_overhead(search_s, self._spans, timing)

        # Model stats, filled in once the best model is built
        report.update(self.parameter_counts(model))
        return report

    @staticmethod
    def parameter_counts(model: Optional[Recommender]) -> Dict[str, float]:
        """The parameter counts of the best model, for the time report.

        Args:
            model (Optional[Recommender]): The best model, None when not built.

        Returns:
            Dict[str, float]: The total and trainable parameter counts, NaN
                without a model.
        """
        if model is None:
            return {
                "Total Params (Best Model)": np.nan,
                "Trainable Params (Best Model)": np.nan,
            }
        return {
            "Total Params (Best Model)": sum(p.numel() for p in model.parameters()),
            "Trainable Params (Best Model)": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
