# pylint: disable=wrong-import-position, wrong-import-order
import argparse
import time
from typing import List, Tuple, Dict, Any
from argparse import Namespace

# Correctly initialize environment variables
from env import initialize_environment

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
initialize_environment(args.config)

import ray
import torch

from warprec.common import (
    initialize_datasets,
    prepare_train_loaders,
    dataset_preparation,
)
from warprec.data.reader import ReaderFactory
from warprec.data.writer import WriterFactory
from warprec.data import Dataset
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
    retrieve_evaluation_dataloader,
)
from warprec.utils.logger import logger
from warprec.recommenders.trainer import Trainer
from warprec.recommenders.loops import train_loop
from warprec.recommenders.base_recommender import (
    Recommender,
    IterativeRecommender,
    SequentialRecommenderUtils,
    ContextRecommenderUtils,
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
    main_dataset, val_dataset, fold_dataset = initialize_datasets(
        reader=reader,
        callback=callback,
        config=config,
    )

    # Write split information if required
    if config.splitter and config.writer.save_split:
        writer.write_split(
            main_dataset,
            val_dataset,
            fold_dataset,
            **config.writer.split.model_dump(),
        )

    # Trainer testing
    models = list(config.models.keys())

    # If statistical significance is required, metrics will
    # be computed user-wise
    requires_stat_significance = (
        config.evaluation.stat_significance.requires_stat_significance()
    )
    if requires_stat_significance:
        logger.attention(
            "Statistical significance is required, metrics will be computed user-wise."
        )
        model_results: Dict[str, Any] = {}

    # General check for user-wise computation
    compute_per_user = config.evaluation.compute_per_user
    requires_per_user_evaluation = requires_stat_significance or compute_per_user

    # Create instance of main evaluator used to evaluate the main dataset
    evaluator = Evaluator(
        list(config.evaluation.metrics),
        list(config.evaluation.top_k),
        train_set=main_dataset.train_set.get_sparse(),
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
        compute_per_user=requires_per_user_evaluation,
        feature_lookup=main_dataset.get_features_lookup(),
        user_cluster=main_dataset.get_user_cluster(),
        item_cluster=main_dataset.get_item_cluster(),
    )

    # Prepare dataloaders for evaluation
    preparation_strategy = config.general.train_data_preparation
    dataset_preparation(main_dataset, fold_dataset, preparation_strategy, config)

    data_preparation_time = time.time() - experiment_start_time
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )
    model_timing_report = []

    # Before starting training process, initialize Ray
    py_modules = (
        [] if config.general.custom_models is None else config.general.custom_models
    )
    py_modules.extend(["warprec"])  # type: ignore[union-attr]
    ray.init(runtime_env={"py_modules": py_modules})

    for model_name in models:
        model_exploration_start_time = time.time()

        # Check if dataloader requirements is in 'model' mode
        if preparation_strategy == "conservative":
            model_dict = {model_name: config.models[model_name]}
            prepare_train_loaders(main_dataset, model_dict)

            for fold in fold_dataset:
                prepare_train_loaders(fold, model_dict)

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

        # Retrieve appropriate evaluation dataloader
        dataloader = retrieve_evaluation_dataloader(
            dataset=main_dataset,
            model=best_model,
            strategy=config.evaluation.strategy,
            num_negatives=config.evaluation.num_negatives,
        )

        # Retrieve best model device
        best_model_device = best_model.device

        # Evaluation testing
        model_evaluation_start_time = time.time()
        evaluator.evaluate(
            model=best_model,
            dataloader=dataloader,
            strategy=config.evaluation.strategy,
            dataset=main_dataset,
            device=str(best_model_device),
            verbose=True,
        )
        results = evaluator.compute_results()
        model_evaluation_total_time = time.time() - model_evaluation_start_time
        evaluator.print_console(results, "Test", config.evaluation.max_metric_per_row)

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
        if compute_per_user:
            i_umap, _ = main_dataset.get_inverse_mappings()
            writer.write_results_per_user(
                results,
                model_name,
                i_umap,
                **config.writer.results.model_dump(),
            )

        # Recommendation
        if params.meta.save_recs:
            writer.write_recs(
                model=best_model,
                dataset=main_dataset,
                **config.writer.recommendation.model_dump(),
            )

        # Save params
        model_params = {model_name: best_model.get_params()}
        writer.write_params(model_params)

        # Model serialization
        if params.meta.save_model:
            writer.write_model(best_model)

        if config.general.time_report:
            # Retrieve dataset information
            info = main_dataset.info()
            n_users = info.get("n_users", None)
            n_items = info.get("n_items", None)
            context_dims = info.get("context_dims", {})

            # Define simple sample to measure prediction time
            n_users_to_predict = min(1000, n_users)
            n_items_to_predict = min(1000, n_items)

            # Create mock data to test model performance during inference
            if isinstance(best_model, SequentialRecommenderUtils):
                max_seq_len = best_model.max_seq_len
            else:
                max_seq_len = 10

            # Create mock data for Context-Aware models
            contexts = None
            if isinstance(best_model, ContextRecommenderUtils):
                model_labels = best_model.context_labels

                if model_labels:
                    ctx_list = []
                    for label in model_labels:
                        dim = context_dims.get(label, 10)
                        c_data = torch.randint(1, dim, (n_users_to_predict,)).to(
                            device=best_model_device
                        )
                        ctx_list.append(c_data)

                    contexts = torch.stack(ctx_list, dim=1)

            # Create mock data to test prediction time
            user_indices = torch.arange(n_users_to_predict).to(device=best_model_device)
            item_indices = torch.randint(
                1, n_items, (n_users_to_predict, n_items_to_predict)
            ).to(device=best_model_device)
            user_seq = torch.randint(1, n_items, (n_users_to_predict, max_seq_len)).to(
                device=best_model_device
            )
            seq_len = torch.randint(1, max_seq_len + 1, (n_users_to_predict,)).to(
                device=best_model_device
            )
            train_sparse = main_dataset.train_set.get_sparse()
            train_batch = train_sparse[user_indices.tolist(), :]

            # Test inference time
            inference_time_start = time.time()
            best_model.predict(
                user_indices=user_indices,
                item_indices=item_indices,
                user_seq=user_seq,
                seq_len=seq_len,
                train_batch=train_batch,
                train_sparse=train_sparse,
                contexts=contexts,
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

            # Clear out the dataset cache if in 'conservative' mode
            if preparation_strategy == "conservative":
                main_dataset.clear_cache()

                for fold in fold_dataset:
                    fold.clear_cache()

                logger.positive("Dataset cache cleared.")

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

    # Check for multi-gpu scenario
    num_gpus = None
    if params.optimization.multi_gpu:
        num_gpus = params.optimization.num_gpus

    # Evaluation on report
    eval_config = config.evaluation
    val_metric, val_k = validation_metric(config.evaluation.validation_metric)
    if eval_config.full_evaluation_on_report:
        metrics = eval_config.metrics
        topk = eval_config.top_k
    else:
        metrics = [val_metric]
        topk = [val_k]

    # Retrieve storage path for Ray results
    # based on the writer configuration
    storage_path = config.get_storage_path()

    # Start HPO phase on test set,
    # no need of further training
    best_model, ray_report = trainer.train_single_fold(
        model_name,
        params,
        dataset,
        metrics=metrics,
        topk=topk,
        validation_score=config.evaluation.validation_metric,
        storage_path=storage_path,
        num_gpus=num_gpus,
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

    # Check for multi-gpu scenario
    num_gpus = None
    if params.optimization.multi_gpu:
        num_gpus = params.optimization.num_gpus

    # Retrieve common params
    block_size = params.optimization.block_size
    chunk_size = params.optimization.chunk_size
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

    # Retrieve storage path for Ray results
    # based on the writer configuration
    storage_path = config.get_storage_path()

    # Start HPO phase on validation folds
    best_params, report = trainer.train_multiple_fold(
        model_name,
        params,
        val_datasets,
        metrics=metrics,
        topk=topk,
        validation_score=validation_score,
        storage_path=storage_path,
        num_gpus=num_gpus,
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
        seed=seed,
        info=main_dataset.info(),
        **main_dataset.get_stash(),
        block_size=block_size,
        chunk_size=chunk_size,
    )
    best_model.to(device)

    # Train the model using backpropagation if the model
    # is iterative
    if isinstance(best_model, IterativeRecommender):
        # Training loop decorated with tqdm for a better visualization
        train_loop(best_model, main_dataset, iterations, device=device)

    # Final reporting
    report["Total Params (Best Model)"] = sum(
        p.numel() for p in best_model.parameters()
    )
    report["Trainable Params (Best Model)"] = sum(
        p.numel() for p in best_model.parameters() if p.requires_grad
    )

    return best_model, report


if __name__ == "__main__":
    main(args)
