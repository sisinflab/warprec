import time
from typing import Dict, Any, Optional, List, Tuple

import ray

from warprec.common import log_evaluation
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
        ray.init(address="auto", runtime_env={"py_modules": py_modules})
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
    writer = WriterFactory.get_writer(config=config)

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

    futures = []
    for model_name in models:
        future = remote_model_pipeline.remote(
            model_name=model_name,
            config=config,
            main_dataset=main_ds_ref,
            val_dataset=val_ds_ref,
            fold_dataset=fold_ds_ref,
            callback=callback,
            data_preparation_time=data_preparation_time,
        )  # type: ignore[call-arg]
        futures.append(future)

    # Wait for all models to finish
    completed_models = ray.get(futures)

    # Final result logging of driver
    logger.msg("Swarming completed. Aggregating and saving results.")

    for (
        model_name,
        best_model,
        results,
        model_params,
        timing_report,
        _,
    ) in completed_models:
        if best_model is None:
            logger.attention(f"HPO for {model_name} returned no valid model.")
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

    # Write aggregated time report (if requested)
    if config.general.time_report and model_timing_report:
        writer.write_time_report(model_timing_report)

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
            corrections = config.evaluation.stat_significance.corrections.model_dump()

            for stat_name, enabled in stat_significance.items():
                if enabled:
                    test_results = compute_paired_statistical_test(
                        model_results, stat_name, **corrections
                    )
                    writer.write_statistical_significance_test(test_results, stat_name)

            logger.positive("Statistical significance tests completed successfully.")
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
    )

    # Run the HPO
    if val_dataset is not None:
        # CASE 2: Train/Validation/Test
        best_model, ray_report, best_iter = single_split_flow(
            model_name, params, val_dataset, trainer, config
        )
    elif len(fold_dataset) > 0:
        # CASE 3: Cross-validation
        best_model, ray_report, best_iter = multiple_fold_validation_flow(
            model_name, params, main_dataset, fold_dataset, trainer, config
        )
    else:
        # CASE 1: Train/Test
        best_model, ray_report, best_iter = single_split_flow(
            model_name, params, main_dataset, trainer, config
        )

    if best_model is None:
        return model_name, None, {}, {}, {}, {}

    model_exploration_total_time = time.time() - model_exploration_start_time

    # Prepare device and resources
    general_device = config.general.device
    model_device = params.optimization.device
    device = general_device if model_device is None else model_device

    num_cpus = params.optimization.cpu_per_trial
    num_gpus = params.optimization.gpu_per_trial
    custom_res = params.optimization.custom_resources_per_trial or {}
    label_selector = params.optimization.label_selector or {}

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
            device=device,
            requires_timing=config.general.time_report,
            custom_modules=config.general.custom_modules,
        )  # type: ignore[call-arg]
    )

    # Recommendation writing
    if params.meta.save_recs:
        from warprec.data.writer import WriterFactory

        writer = WriterFactory.get_writer(config=config)
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
