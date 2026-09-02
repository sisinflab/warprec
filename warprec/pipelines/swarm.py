import time
from typing import Dict, Any, Optional, List, Tuple

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
    remote_evaluation_and_timing,
    remote_generate_recs,
)
from warprec.pipelines.train import single_split_flow, multiple_fold_validation_flow
from warprec.recommenders.trainer import Trainer
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import (
    load_train_configuration,
    load_callback,
    TrainConfiguration,
)
from warprec.utils.pause import PauseController
from warprec.utils.enums import ErroredTrialPolicy, ResumeMode
from warprec.utils.helpers import model_param_from_dict
from warprec.utils.logger import logger
from warprec.evaluation.statistical_significance import compute_paired_statistical_test


def swarm_pipeline(path: str):
    """Main function to start the distributed experiment.

    This method will start the highly parallelized train pipeline,
    launching all models concurrently on the Ray cluster.

    Args:
        path (str): Path to the configuration file.

    Raises:
        ConnectionError: If unable to connect to Ray cluster.
        ValueError: If the file format is not supported.
    """
    logger.attention(
        "WARNING: Swarm pipeline is experimental. Please submit bug reports via GitHub Issues."
    )
    logger.msg("Start experiment swarming.")
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
            pipeline="swarm",
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

    data_preparation_time = time.time() - experiment_start_time
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
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

    # Starting the model swarming
    logger.msg(
        f"Launching the swarm of experiments. Number of different models: {len(models)}"
    )

    # Put datasets in the object store once to avoid sending them multiple times
    main_ds_ref = ray.put(main_dataset)
    val_ds_ref = ray.put(val_dataset)
    fold_ds_ref = ray.put(fold_dataset)

    # Only launch the models that are not already finished
    pending_models = []
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

        current_fingerprint = model_fingerprint(model_name, config.models[model_name])
        if model_state.fingerprint and model_state.fingerprint != current_fingerprint:
            logger.attention(
                f"The configuration of {model_name} changed since the last run. "
                "Its saved state will be discarded and the model will be "
                "optimized from scratch."
            )
            state.models[model_name] = ModelState()
        state.model_state(model_name).fingerprint = current_fingerprint
        pending_models.append(model_name)

    state_store.save(state)

    futures = []
    for model_name in pending_models:
        future = remote_model_pipeline.remote(
            model_name=model_name,
            config=config,
            main_dataset=main_ds_ref,
            val_dataset=val_ds_ref,
            fold_dataset=fold_ds_ref,
            callback=callback,
            data_preparation_time=data_preparation_time,
            run_name=run_name,
            errored_trials=config.run.errored_trials,
            writer_timestamp=state.writer_timestamp,
        )  # type: ignore[call-arg]
        futures.append(future)

    # Wait for the models to finish, staying responsive to a pause request.
    #
    # The tuning loops run inside the worker tasks, so a signal delivered to
    # this driver never reaches Ray Tune's own graceful handler. The best the
    # driver can do is cancel the tasks and rely on Ray Tune's periodic
    # experiment checkpoint, which makes a swarm pause coarser than a train one.
    paused = False
    completed_models = []

    with PauseController(enabled=config.run.pause_on_signal) as pause:
        pending = dict(zip(pending_models, futures))
        while pending:
            if pause.pause_requested:
                paused = True
                logger.attention(
                    "Pause requested. Cancelling the models still running. Their "
                    "Ray Tune experiments will resume from their last saved state."
                )
                for name, future in pending.items():
                    ray.cancel(future, force=False, recursive=True)
                    state.model_state(name).status = ModelStatus.INTERRUPTED
                state_store.save(state)
                break

            ready, _ = ray.wait(list(pending.values()), num_returns=1, timeout=5.0)
            if not ready:
                continue

            finished = ready[0]
            name = next(n for n, f in pending.items() if f == finished)
            del pending[name]
            try:
                completed_models.append(ray.get(finished))
            except (
                ray.exceptions.RayTaskError,
                ray.exceptions.TaskCancelledError,
            ) as e:
                logger.negative(f"Model {name} did not complete: {e}")
                state.model_state(name).status = ModelStatus.INTERRUPTED
                state_store.save(state)

    # Final result logging of driver
    logger.msg("Swarming completed. Aggregating and saving results.")

    for (
        model_name,
        best_model,
        results,
        model_params,
        timing_report,
        ray_report,
    ) in completed_models:
        if best_model is None:
            status = ray_report.get("status", "failed") if ray_report else "failed"
            logger.attention(f"HPO for {model_name} returned no valid model.")
            state.model_state(model_name).status = (
                ModelStatus.INTERRUPTED
                if status == "interrupted"
                else ModelStatus.FAILED
            )
            state_store.save(state)
            continue

        # Callbacks
        callback.on_training_complete(model=best_model)
        callback.on_evaluation_complete(
            model=best_model,
            params=model_params[model_name]["Best Params"],
            results=results,
        )

        # Log the results
        log_evaluation(results, "Test", config.evaluation.max_metric_per_row)

        # Collect for statistical significance
        if requires_stat_significance:
            model_results[model_name] = results

        # Write Results
        writer.write_results(results, model_name, **config.writer.results.model_dump())

        # Write Per-User Results
        if config.evaluation.save_per_user:
            i_umap, _ = main_dataset.get_inverse_mappings()
            writer.write_results_per_user(
                results, model_name, i_umap, **config.writer.results.model_dump()
            )

        # Write Params
        writer.write_params(model_params)

        # Write Model Checkpoint
        if config.models[model_name]["meta"]["save_model"]:
            writer.write_model(best_model)

        # Collect timing report
        if config.general.time_report:
            model_timing_report.append(timing_report)

        # Persist the evaluation results so that a resumed run can still run the
        # paired statistical significance tests over every model.
        if requires_stat_significance:
            state_store.save_eval_results(model_name, results)

        model_state = state.model_state(model_name)
        model_state.status = ModelStatus.COMPLETED
        model_state.timing = timing_report
        state_store.save(state)

    # Write aggregated time report (if requested)
    if config.general.time_report and model_timing_report:
        writer.write_time_report(model_timing_report)

    if paused:
        logger.attention(
            f"Swarm run '{run_name}' has been paused. Progress is saved at "
            f"{state_store.state_path}."
        )
        logger.msg(
            "Resume it by running the same command again with run.name set to "
            f"'{run_name}' and run.resume set to 'auto' or 'force'."
        )
    else:
        # Compute statistical significance (if requested)
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

        logger.positive("Experiment swarming concluded. WarpRec is shutting down.")


@ray.remote(num_cpus=0.05)  # Zero-Resource Orchestrator
def remote_model_pipeline(
    model_name: str,
    config: TrainConfiguration,
    main_dataset: Dataset,
    val_dataset: Optional[Dataset],
    fold_dataset: List[Dataset],
    callback: WarpRecCallback,
    data_preparation_time: float,
    run_name: Optional[str] = None,
    errored_trials: ErroredTrialPolicy = ErroredTrialPolicy.SKIP,
    writer_timestamp: Optional[str] = None,
) -> Tuple[str, Optional[Recommender], Dict, Dict, Dict, Dict]:
    """Orchestrates the entire lifecycle of a single model in parallel.

    This task acts as a lightweight manager. It launches HPO, Retraining,
    Evaluation, and Recommendation generation for a specific model.

    Args:
        model_name (str): The name of the model to process.
        config (TrainConfiguration): The configuration object.
        main_dataset (Dataset): The main dataset.
        val_dataset (Optional[Dataset]): The validation dataset.
        fold_dataset (List[Dataset]): The cross-validation folds.
        callback (WarpRecCallback): The callback object.
        data_preparation_time (float): Time taken for data prep (for reporting).
        run_name (Optional[str]): The identifier of the run, used to give the Ray
            Tune experiment a deterministic name so that it can be restored.
        errored_trials (ErroredTrialPolicy): How to treat trials that errored
            before a pause when restoring the experiment.
        writer_timestamp (Optional[str]): The timestamp pinned into the output
            file names of this run.

    Returns:
        Tuple[str, Optional[Recommender], Dict, Dict, Dict, Dict]: A tuple containing:
            - str: Model name.
            - Optional[Recommender]: Best model trained.
            - results (Dict): Evaluation results.
            - model_params (Dict): Best parameters found.
            - timing_report (Dict): Timing statistics.
            - ray_report (Dict): HPO report.
    """

    model_exploration_start_time = time.time()

    # Retrieve storage path for Ray results
    # based on the writer configuration
    storage_path = config.get_storage_path()

    params = model_param_from_dict(model_name, config.models[model_name])

    trainer = Trainer(
        storage_path=storage_path,
        custom_callback=callback,
        custom_modules=config.general.custom_modules,
        dashboard_config=config.dashboard,
        run_name=run_name,
        errored_trials=errored_trials,
    )

    # Run the HPO
    if val_dataset is not None:
        # CASE 2: Train/Validation/Test
        outcome = single_split_flow(model_name, params, val_dataset, trainer, config)
    elif len(fold_dataset) > 0:
        # CASE 3: Cross-validation
        outcome = multiple_fold_validation_flow(
            model_name, params, main_dataset, fold_dataset, trainer, config
        )
    else:
        # CASE 1: Train/Test
        outcome = single_split_flow(model_name, params, main_dataset, trainer, config)

    if outcome.interrupted or outcome.failed or outcome.model is None:
        # The last element reports the status so that the driver can tell an
        # interrupted model from a failed one.
        return model_name, None, {}, {}, {}, {"status": outcome.status}

    best_model = outcome.model
    ray_report = outcome.report
    best_iter = outcome.best_iter

    model_exploration_total_time = time.time() - model_exploration_start_time

    # Prepare device and resources
    general_device = config.general.device
    model_device = params.optimization.device
    device = general_device if model_device is None else model_device

    num_cpus = params.optimization.cpu_per_trial
    num_gpus = params.optimization.gpu_per_trial
    custom_res = params.optimization.custom_resources_per_trial or {}
    label_selector = params.optimization.label_selector or {}

    # Fallback: in case the device is set to cuda but no GPUs are requested,
    # we set num_gpus to 1 to ensure the correct node is selected
    if device == "cuda" and num_gpus == 0:
        num_gpus = 1

    # Execute evaluation on a proper device
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

    # Recommendation writing
    if params.meta.save_recs:
        from warprec.data.writer import WriterFactory

        writer = WriterFactory.get_writer(config=config, timestamp=writer_timestamp)
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

    # Prepare the report to return to the driver
    model_params = {
        model_name: {
            "Best Params": best_model.get_params(),
            "Best Training Iteration": best_iter,
        }
    }

    timing_report = {
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

    return model_name, best_model, results, model_params, timing_report, ray_report
