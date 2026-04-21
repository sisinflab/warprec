import time
from typing import List, Tuple, Dict, Any

import ray

from warprec.common import log_evaluation
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
from warprec.utils.logger import logger
from warprec.recommenders.trainer import Trainer
from warprec.recommenders.base_recommender import Recommender
from warprec.evaluation.statistical_significance import compute_paired_statistical_test


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

    for model_name in models:
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

        if val_dataset is not None:
            # CASE 2: Train/Validation/Test
            best_model, ray_report, best_iter = single_split_flow(
                model_name, params, val_dataset, trainer, config
            )
        elif len(fold_dataset) > 0:
            # CASE 3: Cross-validation
            best_model, ray_report, best_iter = multiple_fold_validation_flow(
                model_name,
                params,
                main_dataset,
                fold_dataset,
                trainer,
                config,
            )
        else:
            # CASE 1: Train/Test
            best_model, ray_report, best_iter = single_split_flow(
                model_name, params, main_dataset, trainer, config
            )

        if best_model is None:
            logger.attention(
                f"Hyperparameter optimization for {model_name} returned no valid model."
            )
            continue

        model_exploration_total_time = time.time() - model_exploration_start_time

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

        # Offload evaluation to worker node
        logger.msg(f"Delegating evaluation of {model_name} model to Ray cluster")
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
    logger.positive("All experiments concluded. WarpRec is shutting down.")


def single_split_flow(
    model_name: str,
    params: RecomModel,
    dataset: Dataset,
    trainer: Trainer,
    config: TrainConfiguration,
) -> Tuple[Recommender, dict, int]:
    """Hyperparameter optimization over a single split.

    The split can either be train/test or train/validation.

    Args:
        model_name (str): Name of the model to optimize.
        params (RecomModel): The parameter used to train the model.
        dataset (Dataset): The main dataset which represents train/test split.
        trainer (Trainer): The trainer instance used to optimize the model.
        config (TrainConfiguration): The configuration file.

    Returns:
        Tuple[Recommender, dict, int]:
            - Recommender: The best model, validated on the folds and trained on
                the main data split.
            - dict: Report dictionary.
            - int: The best training iteration.
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
    best_model, ray_report, best_iter = trainer.train_single_fold(
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

    return best_model, ray_report, best_iter


def multiple_fold_validation_flow(
    model_name: str,
    params: RecomModel,
    main_dataset: Dataset,
    val_datasets: List[Dataset],
    trainer: Trainer,
    config: TrainConfiguration,
) -> Tuple[Recommender, dict, int]:
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
        Tuple[Recommender, dict, int]:
            - Recommender: The best model, validated on the folds and trained on
                the main data split.
            - dict: Report dictionary.
            - int: The best training iteration.
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
    best_params, report = trainer.train_multiple_fold(
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

    # Check in case the HPO failed
    if best_params is None:
        return None, report, 0

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
    report.update(retrain_report)

    return best_model, report, iterations
