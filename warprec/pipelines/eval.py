# pylint: disable=too-many-branches, too-many-statements
import time
from typing import Dict, Any

import torch

from warprec.common import initialize_datasets
from warprec.data.reader import ReaderFactory
from warprec.data.writer import WriterFactory
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import load_eval_configuration, load_callback
from warprec.utils.helpers import retrieve_evaluation_dataloader, model_param_from_dict
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.evaluation.evaluator import Evaluator
from warprec.evaluation.statistical_significance import compute_paired_statistical_test


def eval_pipeline(path: str):
    """Main function to start the evaluation pipeline.

    During the evaluation execution models are expected
    to be already trained and will only be evaluated.

    Args:
        path (str): Path to the configuration file.
    """
    logger.msg("Starting the Evaluation Pipeline.")
    experiment_start_time = time.time()

    # Configuration loading
    config = load_eval_configuration(path)

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
    main_dataset, _, _ = initialize_datasets(
        reader=reader,
        callback=callback,
        config=config,
    )

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

    # Create instance of main evaluator used to evaluate the main dataset
    evaluator = Evaluator(
        list(config.evaluation.metrics),
        list(config.evaluation.top_k),
        train_set=main_dataset.train_set.get_sparse(),
        additional_data=main_dataset.get_stash(),
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
        feature_lookup=main_dataset.get_features_lookup(),
        user_cluster=main_dataset.get_user_cluster(),
        item_cluster=main_dataset.get_item_cluster(),
    )

    data_preparation_time = time.time() - experiment_start_time
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )

    for model_name, model_params in config.models.items():
        params = model_param_from_dict(model_name, model_params)

        # Evaluation params
        block_size = params.optimization.block_size
        chunk_size = params.optimization.chunk_size

        model = model_registry.get(
            name=model_name,
            params=model_params,
            interactions=main_dataset.train_set,
            seed=42,
            info=main_dataset.info(),
            **main_dataset.get_stash(),
            block_size=block_size,
            chunk_size=chunk_size,
        )

        if isinstance(model, IterativeRecommender):
            if params.meta.load_from is not None:
                # Load model weights
                checkpoint = torch.load(
                    params.meta.load_from, weights_only=False, map_location="cpu"
                )
                model.load_state_dict(checkpoint["state_dict"])
                logger.positive("Successfully loaded model previous checkpoint.")
            else:
                logger.negative(
                    "No checkpoint path found. Model will be evaluated using default weights."
                )

        # Callback on training complete
        callback.on_training_complete(model=model)

        # Retrieve appropriate evaluation dataloader
        dataloader = retrieve_evaluation_dataloader(
            dataset=main_dataset,
            model=model,
            strategy=config.evaluation.strategy,
            num_negatives=config.evaluation.num_negatives,
        )

        # Move model to device
        general_device = config.general.device
        model_device = params.optimization.device
        device = general_device if model_device is None else model_device
        model.to(device)

        # Evaluation on main dataset
        evaluator.evaluate(
            model=model,
            dataloader=dataloader,
            strategy=config.evaluation.strategy,
            dataset=main_dataset,
            device=device,
            verbose=True,
        )
        results = evaluator.compute_results()
        evaluator.print_console(results, "Test", config.evaluation.max_metric_per_row)

        if requires_stat_significance:
            model_results[model_name] = (
                results  # Populate model_results for statistical significance
            )

        # Callback after complete evaluation
        callback.on_evaluation_complete(
            model=model,
            params=model_params,
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

        # Recommendation
        if params.meta.save_recs:
            writer.write_recs(
                model=model,
                dataset=main_dataset,
                **config.writer.recommendation.model_dump(),
            )

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
    logger.positive(
        "Evaluation pipeline executed successfully. WarpRec is shutting down."
    )
