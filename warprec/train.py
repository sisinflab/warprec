import os
import argparse
import time
from typing import List, Tuple, Dict, Any, Optional
from argparse import Namespace

import ray
import torch
from pandas import DataFrame

from warprec.data.reader import ReaderFactory
from warprec.data.writer import WriterFactory
from warprec.data.splitting import Splitter
from warprec.data.dataset import Dataset
from warprec.data.filtering import apply_filtering
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import (
    load_train_configuration,
    load_callback,
    TrainConfiguration,
    RecomModel,
)
from warprec.utils.helpers import model_param_from_dict, validation_metric
from warprec.utils.logger import logger
from warprec.recommenders.trainer import Trainer
from warprec.recommenders.loops import train_loop
from warprec.recommenders.base_recommender import (
    Recommender,
    IterativeRecommender,
    SequentialRecommenderUtils,
)
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
    config = load_train_configuration(args.config)

    # Setup visible devices
    visible_devices = config.general.cuda_visible_devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_devices))  # type: ignore[arg-type]

    # Load custom callback if specified
    callback: WarpRecCallback = load_callback(
        config.general.callback,
        *config.general.callback.args,
        **config.general.callback.kwargs,
    )

    # Initialize I/O modules
    reader = ReaderFactory.get_reader(config=config)
    writer = WriterFactory.get_writer(config=config)

    # Dataset loading
    main_dataset: Dataset = None
    val_data: List[Tuple[DataFrame, DataFrame]] | DataFrame = None
    train_data: DataFrame = None
    test_data: DataFrame = None
    side_data = None
    user_cluster = None
    item_cluster = None
    if config.reader.loading_strategy == "dataset":
        data = reader.read()
        data = callback.on_data_reading(data)

        # Check for optional filtering
        if config.filtering is not None:
            filters = config.get_filters()
            data = apply_filtering(data, filters)

        # Splitter testing
        if config.splitter:
            splitter = Splitter(config)

            if config.reader.data_type == "transaction":
                train_data, val_data, test_data = splitter.split_transaction(data)

            else:
                raise ValueError("Data type not yet supported.")

    elif config.reader.loading_strategy == "split":
        if config.reader.data_type == "transaction":
            train_data, val_data, test_data = reader.read_transaction_split()

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
        "precision": config.general.precision,
    }

    logger.msg("Creating main dataset")
    main_dataset = Dataset(
        train_data,
        test_data,
        **common_params,
    )

    # Handle validation data
    val_dataset = None
    fold_dataset = []
    if val_data is not None:
        if not isinstance(val_data, list):
            # CASE 2: Train/Validation/Test
            logger.msg("Creating validation dataset")
            val_dataset = Dataset(
                train_data,
                val_data,
                evaluation_set="Validation",
                **common_params,
            )
        else:
            # CASE 3: Cross-Validation
            n_folds = len(val_data)
            for idx, fold in enumerate(val_data):
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
        val_dataset=val_dataset,
        validation_folds=fold_dataset,
    )

    if config.splitter and config.writer.save_split:
        writer.write_split(main_dataset, val_dataset, fold_dataset)

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

    # Prepare dataloaders for evaluation
    dataset_preparation(main_dataset, fold_dataset, config)

    data_preparation_time = time.time() - experiment_start_time
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )
    model_timing_report = []
    # Before starting training process, initialize Ray
    ray.init(runtime_env={"py_modules": config.general.custom_models})

    for model_name in models:
        model_exploration_start_time = time.time()

        params = model_param_from_dict(model_name, config.models[model_name])
        trainer = Trainer(
            custom_callback=callback,
            custom_models=config.general.custom_models,
            config=config,
        )

        if val_dataset is not None:
            # CASE 2: Train/Validation/Test
            best_model, ray_report = single_split_flow(
                model_name, params, val_dataset, trainer, config
            )
        elif len(fold_dataset) > 0:
            # CASE 3: Cross-validation
            best_model, ray_report = multiple_fold_validation_flow(
                model_name,
                params,
                main_dataset,
                fold_dataset,
                trainer,
                config,
            )
        else:
            # CASE 1: Train/Test
            best_model, ray_report = single_split_flow(
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
            params=params.model_dump(),
            results=results,
        )

        # Write results of current model
        writer.write_results(
            results,
            model_name,
        )

        # Recommendation
        if params.meta.save_recs:
            writer.write_recs(best_model, main_dataset, config.writer.recommendation.k)

        # Save params
        model_params = {model_name: best_model.get_params()}
        writer.write_params(model_params)

        # Model serialization
        if params.meta.save_model:
            writer.write_model(best_model)

        if config.general.time_report:
            # Retrieve dataset information
            info = main_dataset.info()
            num_users = info.get("users", None)
            num_items = info.get("items", None)

            # Define simple sample to measure prediction time
            num_users_to_predict = min(1000, num_users)
            num_items_to_predict = min(1000, num_items)

            # Create mock data to test model performance during inference
            if isinstance(best_model, SequentialRecommenderUtils):
                max_seq_len = best_model.max_seq_len
            else:
                max_seq_len = 10

            # Retrieve best model device
            best_model_device = best_model._device

            # Create mock data to test prediction time
            user_indices = torch.arange(num_users_to_predict).to(
                device=best_model_device
            )
            item_indices = torch.randint(
                1, num_items, (num_users_to_predict, num_items_to_predict)
            ).to(device=best_model_device)
            user_seq = torch.randint(
                1, num_items, (num_users_to_predict, max_seq_len)
            ).to(device=best_model_device)
            seq_len = torch.randint(1, max_seq_len + 1, (num_users_to_predict,)).to(
                device=best_model_device
            )
            train_sparse = main_dataset.train_set.get_sparse()
            train_batch = train_sparse[user_indices.tolist(), :]

            # Test inference time
            inference_time_start = time.time()
            best_model.predict_sampled(
                user_indices=user_indices,
                item_indices=item_indices,
                user_seq=user_seq,
                seq_len=seq_len,
                train_batch=train_batch,
                train_sparse=train_sparse,
            )
            inference_time = time.time() - inference_time_start

            # Timing report for the current model
            model_timing_report.append(
                {
                    "Model Name": model_name,
                    "Data Preparation Time": data_preparation_time,
                    "Hyperparameter Exploration Time": model_exploration_total_time,
                    **ray_report,
                    "Evaluation Time": model_evaluation_total_time,
                    "Inference Time": inference_time,
                    "Total Time": model_exploration_total_time
                    + model_evaluation_total_time,
                }
            )

            # Update time report
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
    logger.positive("All experiments concluded. WarpRec is shutting down.")


def single_split_flow(
    model_name: str,
    params: RecomModel,
    dataset: Dataset,
    trainer: Trainer,
    config: TrainConfiguration,
) -> Tuple[Recommender, dict]:
    """Hyperparameter optimization over a single split.

    The split can either be train/test or train/validation.

    Args:
        model_name (str): Name of the model to optimize.
        params (RecomModel): The parameter used to train the model.
        dataset (Dataset): The main dataset which represents train/test split.
        trainer (Trainer): The trainer instance used to optimize the model.
        config (TrainConfiguration): The configuration file.

    Returns:
        Tuple[Recommender, dict]:
            - Recommender: The best model, validated on the folds and trained on
                the main data split.
            - dict: Report dictionary.
    """
    # Check for device
    general_device = config.general.device
    model_device = params.optimization.device
    device = general_device if model_device is None else model_device

    # Evaluation on report
    eval_config = config.evaluation
    val_metric, val_k = validation_metric(config.evaluation.validation_metric)
    if eval_config.full_evaluation_on_report:
        metrics = eval_config.metrics
        topk = eval_config.top_k
    else:
        metrics = [val_metric]
        topk = [val_k]

    # Start HPO phase on test set,
    # no need of further training
    best_model, ray_report = trainer.train_single_fold(
        model_name,
        params,
        dataset,
        metrics=metrics,
        topk=topk,
        validation_score=config.evaluation.validation_metric,
        device=device,
        evaluation_strategy=config.evaluation.strategy,
        num_negatives=config.evaluation.num_negatives,
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
        ray_verbose=config.general.ray_verbose,
    )

    return best_model, ray_report


def multiple_fold_validation_flow(
    model_name: str,
    params: RecomModel,
    main_dataset: Dataset,
    val_datasets: List[Dataset],
    trainer: Trainer,
    config: TrainConfiguration,
) -> Tuple[Recommender, dict]:
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
        Tuple[Recommender, dict]:
            - Recommender: The best model, validated on the folds and trained on
                the main data split.
            - dict: Report dictionary.
    """
    # Check for device
    general_device = config.general.device
    model_device = params.optimization.device
    device = general_device if model_device is None else model_device

    # Retrieve common params
    block_size = params.optimization.block_size
    validation_score = config.evaluation.validation_metric
    desired_training_it = params.optimization.properties.desired_training_it
    seed = params.optimization.properties.seed

    # Evaluation on report
    eval_config = config.evaluation
    val_metric, val_k = validation_metric(config.evaluation.validation_metric)
    if eval_config.full_evaluation_on_report:
        metrics = eval_config.metrics
        topk = eval_config.top_k
    else:
        metrics = [val_metric]
        topk = [val_k]

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
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
        desired_training_it=desired_training_it,
        ray_verbose=config.general.ray_verbose,
    )

    # Check in case the HPO failed
    if best_params is None:
        return None, report

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
        **main_dataset.get_stash(),
        block_size=block_size,
    )

    # Train the model using backpropagation if the model
    # is iterative
    if isinstance(best_model, IterativeRecommender):
        # Training loop decorated with tqdm for a better visualization
        train_loop(best_model, main_dataset, iterations)

    # Final reporting
    report["Total Params (Best Model)"] = sum(
        p.numel() for p in best_model.parameters()
    )
    report["Trainable Params (Best Model)"] = sum(
        p.numel() for p in best_model.parameters() if p.requires_grad
    )

    return best_model, report


def dataset_preparation(
    main_dataset: Dataset,
    fold_dataset: Optional[List[Dataset]],
    config: TrainConfiguration,
):
    """This method prepares the dataloaders inside the dataset
    that will be passed to Ray during HPO. It is important to
    precompute these dataloaders before starting the optimization to
    avoid multiple computations of the same dataloader.

    Args:
        main_dataset (Dataset): The main dataset of train/test split.
        fold_dataset (Optional[List[Dataset]]): The list of validation datasets
            of train/val splits.
        config (TrainConfiguration): The configuration file used for the experiment.
    """

    def prepare_evaluation_loaders(dataset: Dataset, device: str):
        """utility function to prepare the evaluation dataloaders
        for a given dataset based on the evaluation strategy.

        Args:
            dataset (Dataset): The dataset to prepare.
            device (str): The device to use.
        """
        if config.evaluation.strategy == "full":
            dataset.get_evaluation_dataloader()
        elif config.evaluation.strategy == "sampled":
            dataset.get_neg_evaluation_dataloader(
                num_negatives=config.evaluation.num_negatives,
                seed=config.evaluation.seed,
            )

    logger.msg("Preparing main dataset inner structures for training and evaluation.")

    device = config.general.device
    prepare_evaluation_loaders(main_dataset, device)
    if fold_dataset is not None and isinstance(fold_dataset, list):
        for i, dataset in enumerate(fold_dataset):
            logger.msg(
                f"Preparing fold dataset {i + 1}/{len(fold_dataset)} inner structures for training and evaluation."
            )
            prepare_evaluation_loaders(dataset, device)

    logger.positive("All dataset inner structures ready.")


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
