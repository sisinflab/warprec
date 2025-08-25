import argparse
from typing import List, Tuple, Dict, Any
import time
from argparse import Namespace

import ray
from pandas import DataFrame

from warprec.data.reader import LocalReader
from warprec.data.writer import LocalWriter
from warprec.data.splitting import Splitter
from warprec.data.dataset import Dataset
from warprec.data.filtering import apply_filtering
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import load_yaml, load_callback, Configuration
from warprec.utils.logger import logger
from warprec.recommenders.trainer import Trainer
from warprec.recommenders.loops import train_loop
from warprec.recommenders.base_recommender import Recommender, IterativeRecommender
from warprec.evaluation.evaluator import Evaluator
from warprec.evaluation.statistical_significance import compute_paired_statistical_test
from warprec.utils.registry import model_registry


def main(args: Namespace):
    """Main function to start the experiment.

    This method will start the train pipeline.
    """
    logger.msg("Starting experiment.")
    experiment_start_time = time.time()

    # Config parser testing
    config = load_yaml(args.config)

    # Load custom callback if specified
    callback: WarpRecCallback = load_callback(
        config.general.callback,
        *config.general.callback.args,
        **config.general.callback.kwargs,
    )

    # Writer module testing
    writer = LocalWriter(config=config)

    # Reader module testing
    reader = LocalReader(config)

    # Dataset loading
    main_dataset: Dataset = None
    fold_data: List[Tuple[DataFrame, DataFrame]] = None
    train_data: DataFrame = None
    test_data: DataFrame = None
    side_data = None
    user_cluster = None
    item_cluster = None
    if config.reader.loading_strategy == "dataset":
        data = reader.read()

        # Check for optional filtering
        if config.filtering is not None:
            filters = config.get_filters()
            data = apply_filtering(data, filters)

        # Splitter testing
        if config.splitter:
            splitter = Splitter(config)

            if config.reader.data_type == "transaction":
                train_data, fold_data, test_data = splitter.split_transaction(data)

            else:
                raise ValueError("Data type not yet supported.")

    elif config.reader.loading_strategy == "split":
        if config.reader.data_type == "transaction":
            train_data, fold_data, test_data = reader.read_transaction_split()

        else:
            raise ValueError("Data type not yet supported.")

    # Side information reading
    if config.reader.side:
        side_data = reader.read_side_information()

    # Cluster information reading
    if config.reader.clustering:
        user_cluster, item_cluster = reader.read_cluster_information()

    # Dataset common information
    common_params = {
        "side_data": side_data,
        "user_cluster": user_cluster,
        "item_cluster": item_cluster,
        "batch_size": config.evaluation.batch_size,
        "rating_type": config.reader.rating_type,
        "rating_label": config.reader.labels.rating_label,
        "timestamp_label": config.reader.labels.timestamp_label,
        "cluster_label": config.reader.labels.cluster_label,
        "need_session_based_information": config.need_session_based_information,
        "precision": config.general.precision,
    }

    logger.msg("Creating main dataset")
    main_dataset = Dataset(
        train_data,
        test_data,
        **common_params,
    )
    fold_dataset = []
    if fold_data is not None:
        n_folds = len(fold_data)
        for idx, fold in enumerate(fold_data):
            logger.msg(f"Creating fold dataset {idx + 1}/{n_folds}")
            val_train, val_set = fold
            fold_dataset.append(
                Dataset(
                    val_train,
                    val_set,
                    evaluation_set="Validation",
                    **common_params,
                )
            )

    # Callback on dataset creation
    callback.on_dataset_creation(
        main_dataset=main_dataset,
        validation_folds=fold_dataset,
    )

    if config.splitter and config.writer.save_split:
        writer.write_split(main_dataset, fold_dataset)

    # Trainer testing
    models = list(config.models.keys())

    # If statistical significance is required, metrics will
    # be computed user_wise
    requires_stat_significance = (
        config.evaluation.stat_significance.requires_stat_significance()
    )
    if requires_stat_significance:
        logger.attention(
            "Statistical significance is required, metrics will be computed user-wise."
        )
        model_results: Dict[str, Any] = {}

    # Create instance of main evaluator used to evaluate the main dataset
    evaluator = Evaluator(
        list(config.evaluation.metrics),
        list(config.evaluation.top_k),
        train_set=main_dataset.train_set.get_sparse(),
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
        compute_per_user=requires_stat_significance,
        feature_lookup=main_dataset.get_features_lookup(),
        user_cluster=main_dataset.get_user_cluster(),
        item_cluster=main_dataset.get_item_cluster(),
    )

    data_preparation_time = time.time() - experiment_start_time
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )
    model_timing_report = []
    # Before starting training process, initialize Ray
    ray.init(runtime_env={"py_modules": config.general.custom_models})

    for model_name in models:
        model_exploration_start_time = time.time()

        params = config.models[model_name]
        val_metric, val_k = config.validation_metric(
            params["optimization"]["validation_metric"]
        )
        trainer = Trainer(
            custom_callback=callback,
            custom_models=config.general.custom_models,
            config=config,
        )

        if len(fold_dataset) > 0:
            # We validate model results of validation data
            # and then finalize results of test
            best_model, ray_report = multiple_fold_validation_flow(
                model_name,
                params,
                val_metric,
                val_k,
                main_dataset,
                fold_dataset,
                trainer,
                config,
            )
        else:
            # Model will be optimized and evaluated on test set
            best_model, ray_report = single_train_test_split_flow(
                model_name, params, val_metric, val_k, main_dataset, trainer, config
            )

        model_exploration_total_time = time.time() - model_exploration_start_time

        # Callback on training complete
        callback.on_training_complete(model=best_model)

        # Evaluation testing
        model_evaluation_start_time = time.time()
        evaluator.evaluate(
            best_model,
            main_dataset,
            device=str(best_model._device),
            strategy=config.evaluation.strategy,
            num_negatives=config.evaluation.num_negatives,
            verbose=True,
        )
        results = evaluator.compute_results()
        model_evaluation_total_time = time.time() - model_evaluation_start_time
        evaluator.print_console(results, "Test", config.evaluation.max_metric_per_row)

        if requires_stat_significance:
            model_results["Test"][model_name] = (
                results  # Populate model_results for statistical significance
            )

        # Callback after complete evaluation
        callback.on_evaluation_complete(
            model=best_model,
            params=params,
            results=results,
        )

        # Write results of current model
        writer.write_results(
            results,
            model_name,
        )

        # Recommendation
        if params["meta"]["save_recs"]:
            recs = best_model.get_recs(
                main_dataset,
                k=config.writer.recommendation.k,
            )
            writer.write_recs(recs, model_name)

        # Save params
        model_params = {model_name: best_model.get_params()}
        writer.write_params(model_params)

        # Model serialization
        if params["meta"]["save_model"]:
            writer.write_model(best_model)

        # Timing report for the current model
        model_timing_report.append(
            {
                "Model_Name": model_name,
                "Data_Preparation_Time": data_preparation_time,
                "Hyperparameter_Exploration_Time": model_exploration_total_time,
                **ray_report,
                "Evaluation_Time": model_evaluation_total_time,
                "Total_Time": model_exploration_total_time
                + model_evaluation_total_time,
            }
        )

    if config.general.time_report:
        writer.write_time_report(model_timing_report)

    if requires_stat_significance:
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
    logger.positive(
        "All models trained and evaluated successfully. WarpRec is shutting down."
    )


def single_train_test_split_flow(
    model_name: str,
    params: dict,
    val_metric: str,
    val_k: int,
    dataset: Dataset,
    trainer: Trainer,
    config: Configuration,
) -> Tuple[Recommender, dict]:
    """Hyperparameter optimization over a single train/test split.

    Args:
        model_name (str): Name of the model to optimize.
        params (dict): The parameter used to train the model.
        val_metric (str): The metric used to validate the model.
        val_k (int): The cutoff used to validate the model.
        dataset (Dataset): The main dataset which represents train/test split.
        trainer (Trainer): The trainer instance used to optimize the model.
        config (Configuration): The configuration file.

    Returns:
        Tuple[Recommender, dict]:
            - Recommender: The best model, validated on the folds and trained on
                the main data split.
            - dict: Report dictionary.
    """
    # Start HPO phase on test set,
    # no need of further training
    best_model, ray_report = trainer.train_single_fold(
        model_name,
        params,
        dataset,
        val_metric,
        val_k,
        evaluation_strategy=config.evaluation.strategy,
        num_negatives=config.evaluation.num_negatives,
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
        ray_verbose=config.general.ray_verbose,
    )

    return best_model, ray_report


def multiple_fold_validation_flow(
    model_name: str,
    params: dict,
    val_metric: str,
    val_k: int,
    main_dataset: Dataset,
    val_datasets: List[Dataset],
    trainer: Trainer,
    config: Configuration,
) -> Tuple[Recommender, dict]:
    """Hyperparameter optimization with cross-validation logic.

    Args:
        model_name (str): Name of the model to optimize.
        params (dict): The parameter used to train the model.
        val_metric (str): The metric used to validate the model.
        val_k (int): The cutoff used to validate the model.
        main_dataset (Dataset): The main dataset which represents train/test split.
        val_datasets (List[Dataset]): The validation datasets which represents train/val splits.
            The list can contain n folds of train/val splits.
        trainer (Trainer): The trainer instance used to optimize the model.
        config (Configuration): The configuration file.

    Returns:
        Tuple[Recommender, dict]:
            - Recommender: The best model, validated on the folds and trained on
                the main data split.
            - dict: Report dictionary.
    """
    # Retrieve common params
    device = params["optimization"]["device"]
    block_size = params["optimization"]["block_size"]
    desired_training_it = params["optimization"]["properties"]["desired_training_it"]
    seed = params["optimization"]["properties"]["seed"]

    # Start HPO phase on validation folds
    best_params, report = trainer.train_multiple_fold(
        model_name,
        params,
        val_datasets,
        val_metric,
        val_k,
        evaluation_strategy=config.evaluation.strategy,
        num_negatives=config.evaluation.num_negatives,
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
        desired_training_it=desired_training_it,
        ray_verbose=config.general.ray_verbose,
    )

    logger.msg(f"Initializing {model_name} model for test set evaluation")

    # Retrieve the model from the registry
    # using the best parameters
    iterations = best_params["iterations"]
    best_model = model_registry.get(
        name=model_name,
        params=best_params,
        interactions=main_dataset.train_set,
        device=device,
        seed=seed,
        info=main_dataset.info(),
        block_size=block_size,
    )

    # Train the model using backpropagation if the model
    # is iterative
    if isinstance(best_model, IterativeRecommender):
        # Training loop decorated with tqdm for a better visualization
        train_loop(best_model, main_dataset, iterations)

    # Final reporting
    report["Total_Params (Best Model)"] = sum(
        p.numel() for p in best_model.parameters()
    )
    report["Trainable_Params (Best Model)"] = sum(
        p.numel() for p in best_model.parameters() if p.requires_grad
    )

    return best_model, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        action="store",
        required=True,
        help="Config file local path",
    )
    args = parser.parse_args()
    main(args)
