import time
from typing import List, Optional, Dict, Any

import ray

from warprec.common import (
    ModelState,
    ModelStatus,
    RunState,
    RunStateStore,
    log_evaluation,
    model_fingerprint,
    resolve_run_name,
    run_fingerprint,
    warprec_version,
)
from warprec.data.reader import ReaderFactory
from warprec.data.writer import WriterFactory
from warprec.data import Dataset
from warprec.pipelines.remotes import (
    remote_data_preparation,
    remote_model_retraining,
    remote_evaluation_and_timing,
    remote_generate_recs,
)
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import (
    load_train_configuration,
    load_callback,
    TrainConfiguration,
    RecomModel,
)
from warprec.utils.helpers import (
    model_param_from_dict,
    validation_metric,
)
from warprec.utils.pause import PauseController, RunPaused
from warprec.utils.enums import ResumeMode
from warprec.utils.logger import logger
from warprec.recommenders.trainer import Trainer, TrainingOutcome
from warprec.evaluation.statistical_significance import compute_paired_statistical_test


def resolve_model_outcome(
    model_name: str,
    params: RecomModel,
    trainer: Trainer,
    config: TrainConfiguration,
    main_dataset: Dataset,
    val_dataset: Optional[Dataset],
    fold_dataset: List[Dataset],
    model_state: ModelState,
) -> TrainingOutcome:
    """Produces the training outcome for one model, reusing saved state if possible.

    When the run state records that hyperparameter optimization already finished
    for this model, the best model is rebuilt from its checkpoint instead of
    running the sweep again. Cross-validated runs produce no single best
    checkpoint, so they fall through to a normal run of the flow.

    Args:
        model_name (str): The name of the model.
        params (RecomModel): The parameters of the model.
        trainer (Trainer): The trainer instance used to optimize the model.
        config (TrainConfiguration): The configuration file.
        main_dataset (Dataset): The main dataset which represents train/test split.
        val_dataset (Optional[Dataset]): The validation dataset, when present.
        fold_dataset (List[Dataset]): The cross-validation folds, when present.
        model_state (ModelState): The saved state of this model.

    Returns:
        TrainingOutcome: The outcome of the optimization for this model.
    """
    if (
        model_state.status == ModelStatus.HPO_COMPLETED
        and model_state.best_checkpoint_path
        and model_state.best_params is not None
    ):
        logger.msg(
            f"Reusing the completed hyperparameter search for {model_name} "
            f"from {model_state.best_checkpoint_path}."
        )
        try:
            model = trainer.load_best_model(
                model_name,
                model_state.best_checkpoint_path,
                model_state.best_params,
                main_dataset,
            )
            return TrainingOutcome(
                status="completed",
                model=model,
                best_params=model_state.best_params,
                report=model_state.ray_report,
                best_iter=model_state.best_iter,
                best_checkpoint_path=model_state.best_checkpoint_path,
            )
        except (FileNotFoundError, OSError, RuntimeError, KeyError) as e:
            logger.attention(
                f"Could not rebuild {model_name} from its saved checkpoint ({e}). "
                "The hyperparameter search will run again."
            )

    if val_dataset is not None:
        # CASE 2: Train/Validation/Test
        return single_split_flow(model_name, params, val_dataset, trainer, config)
    if len(fold_dataset) > 0:
        # CASE 3: Cross-validation
        return multiple_fold_validation_flow(
            model_name, params, main_dataset, fold_dataset, trainer, config
        )
    # CASE 1: Train/Test
    return single_split_flow(model_name, params, main_dataset, trainer, config)


def train_pipeline(path: str):
    """Main function to start the experiment.

    This method will start the train pipeline.

    Args:
        path (str): Path to the configuration file.

    Raises:
        ConnectionError: If unable to connect to Ray cluster.
        ValueError: If the file format is not supported.
    """
    logger.msg("Starting experiment.")
    experiment_start_time = time.time()

    # Parse configuration
    config = load_train_configuration(path)

    # Before starting training process, initialize Ray
    py_modules = (
        [] if config.general.custom_modules is None else config.general.custom_modules
    )
    py_modules.extend(["warprec"])  # type: ignore[union-attr]

    try:
        address = config.general.ray_address
        ray.init(address=address, runtime_env={"py_modules": py_modules})
        logger.positive("Connected to existing Ray cluster.")
    except ConnectionError as e:
        raise ConnectionError(
            "Unable to connect to Ray cluster. Please ensure Ray is running."
        ) from e

    # Load custom callback if specified
    callback: WarpRecCallback = load_callback(
        config.general.callback,
        *config.general.callback.args,
        **config.general.callback.kwargs,
    )

    # Initialize I/O modules
    reader = ReaderFactory.get_reader(config=config)

    # Resolve the identity of this run and load any state a previous,
    # interrupted run of the same experiment left behind.
    run_name = resolve_run_name(config)
    fingerprint = run_fingerprint(config)
    logger.msg(f"Run name: {run_name}")

    probe_writer = WriterFactory.get_writer(config=config)
    state_store = RunStateStore(probe_writer, run_name)
    previous_state = state_store.load()

    if previous_state is not None and config.run.resume == ResumeMode.NEVER:
        logger.attention(
            f"Run state for '{run_name}' exists but resume is set to 'never'. "
            "It will be discarded and the run will start from scratch."
        )
        previous_state = None
    elif (
        previous_state is not None and previous_state.config_fingerprint != fingerprint
    ):
        message = (
            f"Run state for '{run_name}' was produced by a different configuration "
            "(the reader, filtering, splitter, evaluation or the set of models "
            "changed). It cannot be resumed."
        )
        if config.run.resume == ResumeMode.FORCE:
            raise ValueError(message)
        logger.attention(f"{message} The run will start from scratch.")
        previous_state = None

    if previous_state is None and config.run.resume == ResumeMode.FORCE:
        raise ValueError(
            "Resume is set to 'force' but no resumable state was found for run "
            f"'{run_name}' at {state_store.state_path}."
        )

    if previous_state is None:
        state = RunState(
            run_name=run_name,
            pipeline="train",
            warprec_version=warprec_version(),
            writer_timestamp=probe_writer.timestamp,
            config_fingerprint=fingerprint,
        )
    else:
        state = previous_state
        logger.positive(f"Resuming run '{run_name}' created at {state.created_at}.")

    # The writer reuses the timestamp of the original run so that a resumed run
    # keeps merging into the output files that run created.
    writer = WriterFactory.get_writer(config=config, timestamp=state.writer_timestamp)
    state_store = RunStateStore(writer, run_name)
    state_store.save(state)

    # Load datasets using common utility
    logger.msg("Delegating data preparation to Ray cluster")
    cpu_data_prep = config.general.cpu_data_prep
    custom_res_data_prep = config.general.custom_resources_data_prep
    label_selector_data_prep = config.general.label_selector_data_prep
    main_dataset, val_dataset, fold_dataset = ray.get(
        remote_data_preparation.options(
            num_cpus=cpu_data_prep,
            resources=custom_res_data_prep if custom_res_data_prep else None,
            label_selector=label_selector_data_prep
            if label_selector_data_prep
            else None,
        ).remote(
            reader=reader,
            callback=callback,
            config=config,
        )  # type: ignore[call-arg]
    )

    # Write split information if required
    if config.splitter and config.writer.save_split:
        file_format = config.writer.split.file_format

        match file_format:
            case "tabular":
                writer.write_tabular_split(
                    main_dataset,
                    val_dataset,
                    fold_dataset,
                    **config.writer.split.model_dump(),
                )
            case "parquet":
                writer.write_parquet_split(
                    main_dataset,
                    val_dataset,
                    fold_dataset,
                    **config.writer.split.model_dump(),
                )
            case _:
                raise ValueError(f"File format '{file_format}'not supported.")

    # List of models to train
    models = list(config.models.keys())

    # Check if statistical significance is requested
    requires_stat_significance = (
        config.evaluation.stat_significance.requires_stat_significance()
    )
    if requires_stat_significance:
        model_results: Dict[str, Any] = {}

    data_preparation_time = time.time() - experiment_start_time
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )
    model_timing_report = []

    paused = False

    with PauseController(enabled=config.run.pause_on_signal) as pause:
        try:
            for model_name in models:
                model_state = state.model_state(model_name)

                if model_state.status == ModelStatus.COMPLETED:
                    logger.msg(f"Skipping {model_name}: already completed in this run.")
                    if requires_stat_significance:
                        stored_results = state_store.load_eval_results(model_name)
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
                # based on the writer configuration
                storage_path = config.get_storage_path()

                params = model_param_from_dict(model_name, config.models[model_name])

                # A model whose configuration changed cannot reuse its saved state
                current_fingerprint = model_fingerprint(
                    model_name, config.models[model_name]
                )
                if (
                    model_state.fingerprint
                    and model_state.fingerprint != current_fingerprint
                ):
                    logger.attention(
                        f"The configuration of {model_name} changed since the last run. "
                        "Its saved state will be discarded and the model will be "
                        "optimized from scratch."
                    )
                    model_state = ModelState()
                    state.models[model_name] = model_state
                model_state.fingerprint = current_fingerprint

                trainer = Trainer(
                    storage_path=storage_path,
                    custom_callback=callback,
                    custom_modules=config.general.custom_modules,
                    dashboard_config=config.dashboard,
                    run_name=run_name,
                    errored_trials=config.run.errored_trials,
                )
                model_state.tune_experiment_name = trainer.experiment_name(model_name)

                outcome = resolve_model_outcome(
                    model_name,
                    params,
                    trainer,
                    config,
                    main_dataset,
                    val_dataset,
                    fold_dataset,
                    model_state,
                )

                if outcome.interrupted:
                    # An interrupted sweep is a pause, whether the signal was
                    # observed here or inferred from the sweep itself.
                    model_state.status = ModelStatus.INTERRUPTED
                    state_store.save(state)
                    pause.request_pause()
                    pause.check()

                if outcome.failed or outcome.model is None:
                    logger.attention(
                        f"Hyperparameter optimization for {model_name} returned no valid model."
                    )
                    model_state.status = ModelStatus.FAILED
                    state_store.save(state)
                    continue

                best_model = outcome.model
                ray_report = outcome.report
                best_iter = outcome.best_iter

                # Record that the search finished, so that a pause during the stages
                # that follow does not cost the sweep.
                model_state.status = ModelStatus.HPO_COMPLETED
                model_state.best_params = outcome.best_params
                model_state.best_iter = outcome.best_iter
                model_state.best_checkpoint_path = outcome.best_checkpoint_path
                model_state.ray_report = ray_report
                state_store.save(state)

                model_exploration_total_time = (
                    time.time() - model_exploration_start_time
                )

                # Callback on training complete
                callback.on_training_complete(model=best_model)

                # Prepare device for current model
                general_device = config.general.device
                model_device = params.optimization.device
                device = general_device if model_device is None else model_device

                # Retrieve resources and labels to request correct node from cluster
                num_cpus = params.optimization.cpu_per_trial
                num_gpus = params.optimization.gpu_per_trial
                custom_res = params.optimization.custom_resources_per_trial or {}
                label_selector = params.optimization.label_selector or {}

                # Fallback: in case the device is set to cuda but no GPUs are requested,
                # we set num_gpus to 1 to ensure the correct node is selected
                if device == "cuda" and num_gpus == 0:
                    num_gpus = 1

                # Offload evaluation to worker node
                logger.msg(
                    f"Delegating evaluation of {model_name} model to Ray cluster"
                )
                results, model_evaluation_total_time, inference_time = ray.get(
                    remote_evaluation_and_timing.options(
                        num_cpus=num_cpus,
                        num_gpus=num_gpus,
                        resources=custom_res if custom_res else None,
                        label_selector=label_selector if label_selector else None,
                    ).remote(
                        model=best_model,
                        main_dataset=main_dataset,
                        metrics=config.evaluation.metrics,
                        top_k=config.evaluation.top_k,
                        complex_metrics=config.evaluation.complex_metrics,
                        strategy=config.evaluation.strategy,
                        num_negatives=config.evaluation.num_negatives,
                        num_workers=params.optimization.num_workers,
                        device=device,
                        requires_timing=config.general.time_report,
                        custom_modules=config.general.custom_modules,
                    )  # type: ignore[call-arg]
                )

                # Log the results
                log_evaluation(results, "Test", config.evaluation.max_metric_per_row)

                if requires_stat_significance:
                    model_results[model_name] = (
                        results  # Populate model_results for statistical significance
                    )

                # Callback after complete evaluation
                callback.on_evaluation_complete(
                    model=best_model,
                    params=params.model_dump(),
                    results=results,
                )

                # Write results of current model
                writer.write_results(
                    results,
                    model_name,
                    **config.writer.results.model_dump(),
                )

                # Check if per-user results are needed
                if config.evaluation.save_per_user:
                    i_umap, _ = main_dataset.get_inverse_mappings()
                    writer.write_results_per_user(
                        results,
                        model_name,
                        i_umap,
                        **config.writer.results.model_dump(),
                    )

                # Recommendation writing
                if params.meta.save_recs:
                    logger.msg(
                        f"Delegating recommendations generation for {model_name} to Ray cluster"
                    )
                    ray.get(
                        remote_generate_recs.options(
                            num_cpus=num_cpus,
                            num_gpus=num_gpus,
                            resources=custom_res if custom_res else None,
                            label_selector=label_selector if label_selector else None,
                        ).remote(
                            writer=writer,
                            model=best_model,
                            dataset=main_dataset,
                            config=config,
                            device=device,
                        )  # type: ignore[call-arg]
                    )

                # Save params
                model_params = {
                    model_name: {
                        "Best Params": best_model.get_params(),
                        "Best Training Iteration": best_iter,
                    }
                }
                writer.write_params(model_params)

                # Model serialization
                if params.meta.save_model:
                    writer.write_model(best_model)

                if config.general.time_report:
                    # Timing report for the current model
                    model_timing_report.append(
                        {
                            "Model Name": model_name,
                            "Data Preparation Time": data_preparation_time,
                            "Hyperparameter Exploration Time": model_exploration_total_time,
                            **ray_report,
                            "Evaluation Time": model_evaluation_total_time,
                            "Inference Time": inference_time,
                            "Total Time": data_preparation_time
                            + model_exploration_total_time
                            + model_evaluation_total_time,
                        }
                    )

                    # Update time report
                    writer.write_time_report(model_timing_report)

                if requires_stat_significance:
                    state_store.save_eval_results(model_name, results)

                model_state.status = ModelStatus.COMPLETED
                model_state.timing = (
                    model_timing_report[-1] if model_timing_report else {}
                )
                state_store.save(state)

                pause.check()

        except RunPaused:
            paused = True
            state_store.save(state)

    if paused:
        logger.attention(
            f"Run '{run_name}' has been paused. Progress is saved at "
            f"{state_store.state_path}."
        )
        logger.msg(
            "Resume it by running the same command again with run.name set to "
            f"'{run_name}' and run.resume set to 'auto' or 'force'."
        )
    else:
        if requires_stat_significance:
            # Check if enough models have been evaluated
            if len(model_results) >= 2:
                logger.msg(
                    f"Computing statistical significance tests for {len(models)} models."
                )

                stat_significance = config.evaluation.stat_significance.model_dump(
                    exclude=["corrections"]  # type: ignore[arg-type]
                )
                corrections = (
                    config.evaluation.stat_significance.corrections.model_dump()
                )

                for stat_name, enabled in stat_significance.items():
                    if enabled:
                        test_results = compute_paired_statistical_test(
                            model_results, stat_name, **corrections
                        )
                        writer.write_statistical_significance_test(
                            test_results, stat_name
                        )

                logger.positive(
                    "Statistical significance tests completed successfully."
                )
            else:
                logger.attention(
                    "Statistical significance tests require at least two evaluated models. "
                    "Skipping statistical significance computation."
                )
        logger.positive("All experiments concluded. WarpRec is shutting down.")


def single_split_flow(
    model_name: str,
    params: RecomModel,
    dataset: Dataset,
    trainer: Trainer,
    config: TrainConfiguration,
) -> TrainingOutcome:
    """Hyperparameter optimization over a single split.

    The split can either be train/test or train/validation.

    Args:
        model_name (str): Name of the model to optimize.
        params (RecomModel): The parameter used to train the model.
        dataset (Dataset): The main dataset which represents train/test split.
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
    if eval_config.full_evaluation_on_report:
        metrics = eval_config.metrics
        topk = eval_config.top_k
        complex_metrics = eval_config.complex_metrics
    else:
        metrics = [val_metric]
        topk = [val_k]
        complex_metrics = []

    # Start HPO phase on test set,
    # no need of further training
    return trainer.train_single_fold(
        model_name,
        params,
        dataset,
        metrics=metrics,
        topk=topk,
        validation_score=validation_score,
        device=device,
        evaluation_strategy=config.evaluation.strategy,
        num_negatives=config.evaluation.num_negatives,
        complex_metrics=complex_metrics,
        ray_verbose=config.general.ray_verbose,
    )


def multiple_fold_validation_flow(
    model_name: str,
    params: RecomModel,
    main_dataset: Dataset,
    val_datasets: List[Dataset],
    trainer: Trainer,
    config: TrainConfiguration,
) -> TrainingOutcome:
    """Hyperparameter optimization with cross-validation logic.

    Args:
        model_name (str): Name of the model to optimize.
        params (RecomModel): The parameter used to train the model.
        main_dataset (Dataset): The main dataset which represents train/test split.
        val_datasets (List[Dataset]): The validation datasets which represents train/val splits.
            The list can contain n folds of train/val splits.
        trainer (Trainer): The trainer instance used to optimize the model.
        config (TrainConfiguration): The configuration file.

    Returns:
        TrainingOutcome: The outcome of the optimization for this model, carrying
            the model retrained on the main data split.
    """
    # Check for device
    general_device = config.general.device
    model_device = params.optimization.device
    device = general_device if model_device is None else model_device

    # Retrieve common params
    validation_score = config.evaluation.validation_metric
    desired_training_it = params.optimization.properties.desired_training_it
    seed = params.optimization.properties.seed

    # Evaluation on report
    eval_config = config.evaluation
    val_metric, val_k = validation_metric(validation_score)
    logger.attention(
        f"Validation metric for this experiment has been set to: {validation_score}"
    )
    if eval_config.full_evaluation_on_report:
        metrics = eval_config.metrics
        topk = eval_config.top_k
        complex_metrics = eval_config.complex_metrics
    else:
        metrics = [val_metric]
        topk = [val_k]
        complex_metrics = []

    # Start HPO phase on validation folds
    outcome = trainer.train_multiple_fold(
        model_name,
        params,
        val_datasets,
        metrics=metrics,
        topk=topk,
        validation_score=validation_score,
        device=device,
        evaluation_strategy=config.evaluation.strategy,
        num_negatives=config.evaluation.num_negatives,
        complex_metrics=complex_metrics,
        desired_training_it=desired_training_it,
        ray_verbose=config.general.ray_verbose,
    )

    # Check in case the HPO was paused or failed
    if outcome.interrupted or outcome.failed:
        return outcome

    best_params = outcome.best_params

    logger.msg(f"Delegating {model_name} model retraining to Ray cluster")

    # Retrieve resources to request correct node from cluster
    num_cpus = params.optimization.cpu_per_trial
    num_gpus = params.optimization.gpu_per_trial
    custom_res = params.optimization.custom_resources_per_trial or {}
    label_selector = params.optimization.label_selector or {}

    # Offload the retraining to a worker node
    best_model, retrain_report, iterations = ray.get(
        remote_model_retraining.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            resources=custom_res if custom_res else None,
            label_selector=label_selector if label_selector else None,
        ).remote(
            model_name=model_name,
            best_params=best_params,
            main_dataset=main_dataset,
            params=params,
            custom_modules=config.general.custom_modules,
            device=device,
            seed=seed,
        )  # type: ignore[call-arg]
    )

    # Merge the parameter counts into the main report
    report = dict(outcome.report)
    report.update(retrain_report)

    return TrainingOutcome(
        status="completed",
        model=best_model,
        best_params=best_params,
        report=report,
        best_iter=iterations,
    )
