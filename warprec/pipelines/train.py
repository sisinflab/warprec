import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Dict, Any

import ray

from warprec.common import (
    ModelState,
    ModelStatus,
    log_evaluation,
    reconcile_model_state,
)
from warprec.data.writer.base_writer import Writer
from warprec.pipelines.common import (
    PipelineContext,
    PreparedDatasets,
    bootstrap_pipeline,
    callback_needs_model,
    data_preparation_seconds,
    measure_emissions,
    model_task_options,
    model_timing_row,
    prepare_datasets,
    raise_failed_loads,
    report_statistical_significance,
    write_model_outputs,
)
from warprec.pipelines.remotes import ModelCompletion, remote_complete_model
from warprec.utils.config import (
    TrainConfiguration,
    RecomModel,
)
from warprec.utils.helpers import (
    model_param_from_dict,
    validation_metric,
)
from warprec.utils.pause import PauseController, RunPaused
from warprec.utils.logger import logger
from warprec.recommenders.trainer import Trainer, TrainingOutcome

if TYPE_CHECKING:
    from warprec.recommenders.trainer.emissions import EmissionsSettings


def resolve_model_outcome(
    model_name: str,
    params: RecomModel,
    trainer: Trainer,
    config: TrainConfiguration,
    data: PreparedDatasets,
    model_state: ModelState,
) -> TrainingOutcome:
    """Produces the training outcome for one model, reusing saved state if possible.

    When the run state records that hyperparameter optimization already finished
    for this model, its result is reused instead of running the sweep again,
    whether or not its best model could be built then. The outcome carries no
    model: 'complete_model' builds it on the cluster.

    Args:
        model_name (str): The name of the model.
        params (RecomModel): The parameters of the model.
        trainer (Trainer): The trainer instance used to optimize the model.
        config (TrainConfiguration): The configuration file.
        data (PreparedDatasets): The datasets of the run, by reference.
        model_state (ModelState): The saved state of this model.

    Returns:
        TrainingOutcome: The outcome of the optimization for this model.
    """
    if (
        model_state.status in (ModelStatus.HPO_COMPLETED, ModelStatus.FAILED_LOAD)
        and model_state.best_params is not None
    ):
        # A cross-validated search keeps no checkpoint: its best model is
        # retrained from the parameters
        checkpoint = model_state.best_checkpoint_path
        if data.cross_validated or (checkpoint and _checkpoint_exists(checkpoint)):
            logger.msg(f"Reusing the completed hyperparameter search for {model_name}.")
            return TrainingOutcome(
                status="completed",
                best_params=model_state.best_params,
                report=model_state.ray_report,
                best_iter=model_state.best_iter,
                best_checkpoint_path=checkpoint,
            )
        logger.attention(
            f"The best checkpoint of {model_name}, {checkpoint}, no longer exists. "
            "The hyperparameter search will run again."
        )

    if data.layout.has_validation:
        # CASE 2: Train/Validation/Test
        return single_split_flow(model_name, params, data.validation, trainer, config)
    if data.cross_validated:
        # CASE 3: Cross-validation
        return multiple_fold_validation_flow(
            model_name, params, data.folds, data.layout.n_folds, trainer, config
        )
    # CASE 1: Train/Test
    return single_split_flow(model_name, params, data.main, trainer, config)


def _checkpoint_exists(checkpoint_path: str) -> bool:
    """Whether a trial checkpoint is still where the search left it.

    A closed-form model has none: it is refitted from its parameters.

    Args:
        checkpoint_path (str): The checkpoint directory, or 'None'.

    Returns:
        bool: True when there is nothing to find or it was found.
    """
    return checkpoint_path == "None" or Path(checkpoint_path).exists()


def complete_model(
    model_name: str,
    params: RecomModel,
    outcome: TrainingOutcome,
    data: PreparedDatasets,
    config: TrainConfiguration,
    writer: Writer,
    return_model: bool = False,
) -> ModelCompletion:
    """Builds, evaluates and writes the best model of a finished search, on the cluster.

    The best model is loaded from its checkpoint, refitted when it is
    closed-form, or retrained on the main split after cross-validation, by a
    task with the resources of one trial, which then evaluates it and writes it.

    Args:
        model_name (str): The name of the model.
        params (RecomModel): The parameters of the model.
        outcome (TrainingOutcome): The completed outcome of its search.
        data (PreparedDatasets): The datasets of the run, by reference.
        config (TrainConfiguration): The configuration file.
        writer (Writer): The writer of the run.
        return_model (bool): Whether to bring the model back, for a callback.

    Returns:
        ModelCompletion: What the task returned.
    """
    device, options = model_task_options(params, config)
    logger.msg(f"Delegating the best {model_name} model to the Ray cluster.")
    return ray.get(
        remote_complete_model.options(**options).remote(
            model_name=model_name,
            params=params,
            outcome=outcome,
            main_dataset=data.main,
            config=config,
            writer=writer,
            cross_validated=data.cross_validated,
            device=device,
            return_model=return_model,
        )  # type: ignore[call-arg]
    )


def train_pipeline(path: str):
    """Main function to start the experiment.

    This method will start the train pipeline.

    Args:
        path (str): Path to the configuration file.
    """
    context = bootstrap_pipeline(path, "train", "Starting experiment.")
    with measure_emissions(context) as emissions:
        _train_models(context, emissions)


def _train_models(  # noqa: PLR0912, PLR0915
    context: PipelineContext, emissions: Optional["EmissionsSettings"]
) -> None:
    """Prepare the data, then optimize, evaluate and write every model.

    Args:
        context (PipelineContext): The context produced by the bootstrap.
        emissions (Optional[EmissionsSettings]): The run's CodeCarbon
            measurement, or None when the run is not measured.
    """
    config = context.config

    # Build the datasets on the cluster and persist the splits if asked to
    data = prepare_datasets(context)
    data_times = data.layout.times
    needs_model = callback_needs_model(context.callback)

    # List of models to train
    models = list(config.models.keys())

    # Check if statistical significance is requested
    requires_stat_significance = (
        config.evaluation.stat_significance.requires_stat_significance()
    )
    if requires_stat_significance:
        model_results: Dict[str, Any] = {}

    data_preparation_time = data_preparation_seconds(context.started_at, emissions)
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )
    data_times.log()
    model_timing_report = []

    paused = False
    failed_loads: List[str] = []

    with PauseController(enabled=config.run.pause_on_signal) as pause:
        try:
            for model_name in models:
                # Compared before the status is read, so that a model whose
                # configuration changed is optimized again whatever its status
                model_state, resumable = reconcile_model_state(
                    context.state, model_name, config.models[model_name]
                )

                if model_state.status == ModelStatus.COMPLETED:
                    logger.msg(f"Skipping {model_name}: already completed in this run.")
                    if requires_stat_significance:
                        stored_results = context.state_store.load_eval_results(
                            model_name
                        )
                        if stored_results is not None:
                            model_results[model_name] = stored_results
                        else:
                            logger.attention(
                                f"Evaluation results for {model_name} could not be loaded. "
                                "It will be excluded from the statistical significance tests."
                            )
                    continue

                if model_state.status == ModelStatus.FAILED:
                    logger.msg(f"Skipping {model_name}: it failed in a previous run.")
                    continue

                pause.check()

                model_exploration_start_time = time.time()

                # Retrieve storage path for Ray results
                # based on the context.writer configuration
                storage_path = config.get_storage_path()

                params = model_param_from_dict(model_name, config.models[model_name])

                trainer = Trainer(
                    storage_path=storage_path,
                    custom_callback=context.callback,
                    custom_modules=config.general.custom_modules,
                    dashboard_config=config.dashboard,
                    run_name=context.run_name,
                    errored_trials=config.run.errored_trials,
                    resumable=resumable,
                    emissions=emissions,
                )
                model_state.tune_experiment_name = trainer.experiment_name(model_name)

                # Recorded before the sweep starts, so that a run killed during
                # it can still continue the sweep under the same configuration
                context.state_store.save(context.state)

                outcome = resolve_model_outcome(
                    model_name,
                    params,
                    trainer,
                    config,
                    data,
                    model_state,
                )

                if outcome.interrupted:
                    # An interrupted sweep is a pause, whether the signal was
                    # observed here or inferred from the sweep itself.
                    model_state.status = ModelStatus.INTERRUPTED
                    context.state_store.save(context.state)
                    pause.request_pause()
                    pause.check()

                if outcome.failed:
                    logger.attention(
                        f"Hyperparameter optimization for {model_name} returned no valid model."
                    )
                    model_state.status = ModelStatus.FAILED
                    context.state_store.save(context.state)
                    continue

                # Record that the search finished before its best model is built,
                # so that neither a pause nor a failure to build it costs the sweep
                model_state.status = ModelStatus.HPO_COMPLETED
                model_state.best_params = outcome.best_params
                model_state.best_iter = outcome.best_iter
                model_state.best_checkpoint_path = outcome.best_checkpoint_path
                model_state.ray_report = outcome.report
                context.state_store.save(context.state)

                search_seconds = time.time() - model_exploration_start_time
                completion = complete_model(
                    model_name,
                    params,
                    outcome,
                    data,
                    config,
                    context.writer,
                    return_model=needs_model,
                )
                if completion.error is not None:
                    logger.negative(
                        f"The best model of {model_name} could not be built: "
                        f"{completion.error}"
                    )
                    model_state.status = ModelStatus.FAILED_LOAD
                    context.state_store.save(context.state)
                    failed_loads.append(model_name)
                    continue

                ray_report = completion.outcome.report
                results = completion.results
                model_exploration_total_time = search_seconds + completion.build_seconds

                # Callback on training complete
                if needs_model:
                    context.callback.on_training_complete(model=completion.model)

                # Log the results
                log_evaluation(results, "Test", config.evaluation.max_metric_per_row)

                if requires_stat_significance:
                    model_results[model_name] = (
                        results  # Populate model_results for statistical significance
                    )

                # Callback after complete evaluation
                if needs_model:
                    context.callback.on_evaluation_complete(
                        model=completion.model,
                        params=params.model_dump(),
                        results=results,
                    )

                # The results and parameters; the model, the per-user results and
                # the recommendations were written on the cluster
                write_model_outputs(
                    context,
                    model_name,
                    results,
                    completion.best_params,
                    completion.outcome.best_iter,
                )

                if config.general.time_report:
                    # Timing report for the current model
                    model_timing_report.append(
                        model_timing_row(
                            model_name=model_name,
                            data_preparation_time=data_preparation_time,
                            exploration_time=model_exploration_total_time,
                            ray_report=ray_report,
                            evaluation_time=completion.evaluation_seconds,
                            inference_time=completion.inference_seconds,
                            data_times=data_times,
                        )
                    )

                    # Update time report
                    context.writer.write_time_report(model_timing_report)

                if requires_stat_significance:
                    context.state_store.save_eval_results(model_name, results)

                model_state.status = ModelStatus.COMPLETED
                model_state.timing = (
                    model_timing_report[-1] if model_timing_report else {}
                )
                context.state_store.save(context.state)

                pause.check()

        except RunPaused:
            paused = True
            context.state_store.save(context.state)

    if paused:
        logger.attention(
            f"Run '{context.run_name}' has been paused. Progress is saved at "
            f"{context.state_store.state_path}."
        )
        logger.msg(
            "Resume it by running the same command again with run.name set to "
            f"'{context.run_name}' and run.resume set to 'auto' or 'force'."
        )
    else:
        if requires_stat_significance:
            report_statistical_significance(context, model_results, models)
        logger.positive("All experiments concluded. WarpRec is shutting down.")

    raise_failed_loads(failed_loads, context.run_name)


def single_split_flow(
    model_name: str,
    params: RecomModel,
    dataset: ray.ObjectRef,
    trainer: Trainer,
    config: TrainConfiguration,
) -> TrainingOutcome:
    """Hyperparameter optimization over a single split.

    The split can either be train/test or train/validation.

    Args:
        model_name (str): Name of the model to optimize.
        params (RecomModel): The parameter used to train the model.
        dataset (ray.ObjectRef): The split to search on, train/test or
            train/validation, in the object store.
        trainer (Trainer): The trainer instance used to optimize the model.
        config (TrainConfiguration): The configuration file.

    Returns:
        TrainingOutcome: The outcome of the optimization for this model.
    """
    # Check for device
    general_device = config.general.device
    model_device = params.optimization.device
    device = general_device if model_device is None else model_device

    # Evaluation on report
    eval_config = config.evaluation
    validation_score = config.evaluation.validation_metric
    val_metric, val_k = validation_metric(validation_score)
    logger.attention(
        f"Validation metric for this experiment has been set to: {validation_score}"
    )
    # Per-report evaluation is narrowed to the validation metric unless the
    # configuration asks for the full set on every report.
    report_evaluation = (
        eval_config
        if eval_config.full_evaluation_on_report
        else eval_config.model_copy(
            update={"metrics": [val_metric], "top_k": [val_k], "complex_metrics": []}
        )
    )

    # Start HPO phase on test set,
    # no need of further training
    return trainer.train_single_fold(
        model_name,
        params,
        dataset,
        evaluation=report_evaluation,
        validation_score=validation_score,
        device=device,
        ray_verbose=config.general.ray_verbose,
    )


def multiple_fold_validation_flow(
    model_name: str,
    params: RecomModel,
    val_datasets: ray.ObjectRef,
    num_folds: int,
    trainer: Trainer,
    config: TrainConfiguration,
) -> TrainingOutcome:
    """Hyperparameter optimization with cross-validation logic.

    Args:
        model_name (str): Name of the model to optimize.
        params (RecomModel): The parameter used to train the model.
        val_datasets (ray.ObjectRef): The list of train/validation folds, in the
            object store.
        num_folds (int): The number of folds.
        trainer (Trainer): The trainer instance used to optimize the model.
        config (TrainConfiguration): The configuration file.

    Returns:
        TrainingOutcome: The outcome of the optimization for this model, without
            a model: 'complete_model' retrains it on the main data split.
    """
    # Check for device
    general_device = config.general.device
    model_device = params.optimization.device
    device = general_device if model_device is None else model_device

    # Retrieve common params
    validation_score = config.evaluation.validation_metric
    desired_training_it = params.optimization.properties.desired_training_it

    # Evaluation on report
    eval_config = config.evaluation
    val_metric, val_k = validation_metric(validation_score)
    logger.attention(
        f"Validation metric for this experiment has been set to: {validation_score}"
    )
    # Per-report evaluation is narrowed to the validation metric unless the
    # configuration asks for the full set on every report.
    report_evaluation = (
        eval_config
        if eval_config.full_evaluation_on_report
        else eval_config.model_copy(
            update={"metrics": [val_metric], "top_k": [val_k], "complex_metrics": []}
        )
    )

    # Start HPO phase on validation folds
    return trainer.train_multiple_fold(
        model_name,
        params,
        val_datasets,
        evaluation=report_evaluation,
        validation_score=validation_score,
        device=device,
        desired_training_it=desired_training_it,
        ray_verbose=config.general.ray_verbose,
        num_folds=num_folds,
    )
