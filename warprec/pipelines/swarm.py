import time
from typing import Dict, Any, Optional, List, Tuple

import ray

from warprec.common import (
    ModelState,
    ModelStatus,
    log_evaluation,
    model_fingerprint,
)
from warprec.data.writer import WriterFactory
from warprec.data import Dataset
from warprec.pipelines.common import (
    bootstrap_pipeline,
    prepare_datasets,
    report_statistical_significance,
)
from warprec.pipelines.remotes import (
    remote_evaluation_and_timing,
    remote_generate_recs,
)
from warprec.pipelines.train import single_split_flow, multiple_fold_validation_flow
from warprec.recommenders.trainer import Trainer
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import (
    TrainConfiguration,
)
from warprec.utils.pause import PauseController
from warprec.utils.enums import ErroredTrialPolicy
from warprec.utils.helpers import model_param_from_dict
from warprec.utils.logger import logger


def swarm_pipeline(path: str):
    """Main function to start the distributed experiment.

    This method will start the highly parallelized train pipeline,
    launching all models concurrently on the Ray cluster.

    Args:
        path (str): Path to the configuration file.
    """
    logger.attention(
        "WARNING: Swarm pipeline is experimental. Please submit bug reports via GitHub Issues."
    )
    context = bootstrap_pipeline(path, "swarm", "Start experiment swarming.")
    config = context.config

    # Build the datasets on the cluster and persist the splits if asked to
    main_dataset, val_dataset, fold_dataset = prepare_datasets(context)

    # List of models to train
    models = list(config.models.keys())

    # Check if statistical significance is requested
    requires_stat_significance = (
        config.evaluation.stat_significance.requires_stat_significance()
    )
    if requires_stat_significance:
        model_results: Dict[str, Any] = {}

    data_preparation_time = time.time() - context.started_at
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )
    model_timing_report: List[Dict[str, Any]] = []

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
        model_state = context.state.model_state(model_name)

        if model_state.status == ModelStatus.COMPLETED:
            logger.msg(f"Skipping {model_name}: already completed in this run.")
            if requires_stat_significance:
                stored_results = context.state_store.load_eval_results(model_name)
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
                "Its saved context.state will be discarded and the model will be "
                "optimized from scratch."
            )
            context.state.models[model_name] = ModelState()
        context.state.model_state(model_name).fingerprint = current_fingerprint
        pending_models.append(model_name)

    context.state_store.save(context.state)

    futures = []
    for model_name in pending_models:
        future = remote_model_pipeline.remote(
            model_name=model_name,
            config=config,
            main_dataset=main_ds_ref,
            val_dataset=val_ds_ref,
            fold_dataset=fold_ds_ref,
            callback=context.callback,
            data_preparation_time=data_preparation_time,
            run_name=context.run_name,
            errored_trials=config.run.errored_trials,
            writer_timestamp=context.state.writer_timestamp,
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
                    "Ray Tune experiments will resume from their last saved context.state."
                )
                for name, future in pending.items():
                    ray.cancel(future, force=False, recursive=True)
                    context.state.model_state(name).status = ModelStatus.INTERRUPTED
                context.state_store.save(context.state)
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
                context.state.model_state(name).status = ModelStatus.INTERRUPTED
                context.state_store.save(context.state)

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
            context.state.model_state(model_name).status = (
                ModelStatus.INTERRUPTED
                if status == "interrupted"
                else ModelStatus.FAILED
            )
            context.state_store.save(context.state)
            continue

        # Callbacks
        context.callback.on_training_complete(model=best_model)
        context.callback.on_evaluation_complete(
            model=best_model,
            params=model_params[model_name]["Best Params"],
            results=results,
        )

        # Log the results
        log_evaluation(results, "Test", config.evaluation.max_metric_per_row)

        # Collect for statistical significance
        if requires_stat_significance:
            context.state_store.save_eval_results(model_name, results)

        model_state = context.state.model_state(model_name)
        model_state.status = ModelStatus.COMPLETED
        model_state.timing = timing_report
        context.state_store.save(context.state)

    # Write aggregated time report (if requested)
    if config.general.time_report and model_timing_report:
        context.writer.write_time_report(model_timing_report)

    if paused:
        logger.attention(
            f"Swarm run '{context.run_name}' has been paused. Progress is saved at "
            f"{context.state_store.state_path}."
        )
        logger.msg(
            "Resume it by running the same command again with run.name set to "
            f"'{context.run_name}' and run.resume set to 'auto' or 'force'."
        )
    else:
        # Compute statistical significance (if requested)
        if requires_stat_significance:
            report_statistical_significance(context, model_results, models)

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
            mask_seen=config.evaluation.mask_seen,
            num_workers=params.optimization.num_workers,
            device=device,
            requires_timing=config.general.time_report,
            custom_modules=config.general.custom_modules,
        )  # type: ignore[call-arg]
    )

    # Recommendation writing
    if params.meta.save_recs:
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
