import os
import time
from typing import Dict, Any

import torch

from warprec.common import initialize_datasets, log_evaluation
from warprec.data import Dataset
from warprec.data.reader import ReaderFactory
from warprec.data.writer import WriterFactory
from warprec.utils.callback import WarpRecCallback
from warprec.evaluation import build_evaluator
from warprec.recommenders.reranking import build_reranker
from warprec.utils.config import load_eval_configuration, load_callback
from warprec.utils.helpers import (
    build_evaluation_dataloader_kwargs,
    resolve_available_cpus,
    resolve_num_workers,
    retrieve_evaluation_dataloader,
    model_param_from_dict,
)
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry
from warprec.recommenders.base_recommender import IterativeRecommender, Recommender
from warprec.evaluation.statistical_significance import compute_paired_statistical_test


def _check_checkpoint_paths(models: Dict[str, Any]) -> None:
    """Refuse a configuration whose checkpoints are not there.

    Args:
        models (Dict[str, Any]): The configured models and their parameters.

    Raises:
        FileNotFoundError: If a model's meta.load_from names a file that does
            not exist.
    """
    for model_name, model_params in models.items():
        load_from = model_param_from_dict(model_name, model_params).meta.load_from
        if load_from is not None and not os.path.isfile(load_from):
            raise FileNotFoundError(
                f"The checkpoint of {model_name} (meta.load_from) does not "
                f"exist: {load_from}"
            )


def _check_checkpoint_matches(
    checkpoint: Dict[str, Any], model_class: type, dataset: Dataset, path: str
) -> None:
    """Refuse a checkpoint that would score the wrong model or the wrong ids.

    A model scores indices, not ids. A checkpoint trained on a dataset whose
    ids map to other indices - another split, another filtering, other data -
    has the same dimensions as this one often enough, and would then rank the
    wrong items for the wrong users without any error.

    Args:
        checkpoint (Dict[str, Any]): The loaded checkpoint.
        model_class (type): The class of the configured model.
        dataset (Dataset): The dataset the model is evaluated on.
        path (str): Where the checkpoint was read from.

    Raises:
        ValueError: If the checkpoint is of another model, or its user or item
            ids do not map to the same indices as the dataset's.
    """
    saved_name = checkpoint.get("name")
    if saved_name != model_class.__name__:
        raise ValueError(
            f"The checkpoint {path} holds a {saved_name} model, not a "
            f"{model_class.__name__}."
        )

    saved_info = checkpoint.get("info") or {}
    info = dataset.info()
    for kind in ("user", "item"):
        saved = saved_info.get(f"{kind}_mapping")
        if saved is None:
            logger.attention(
                f"The checkpoint {path} does not record its {kind} ids, so "
                "they cannot be checked against the dataset."
            )
            continue
        current = info[f"{kind}_mapping"]
        if dict(saved) == dict(current):
            continue
        moved = sum(1 for label, idx in current.items() if saved.get(label) != idx)
        raise ValueError(
            f"The checkpoint {path} was trained on other {kind} ids than this "
            f"dataset: {len(saved)} {kind}s in the checkpoint, {len(current)} in "
            f"the dataset, {moved} of the dataset's {kind}s missing from the "
            "checkpoint or at another index. Evaluate it on the data, filtering "
            "and splitting it was trained with, validation_splitting included: "
            "a model trained with a validation split was fitted without it."
        )


def _load_or_build_model(
    model_name: str,
    model_params: Dict[str, Any],
    dataset: Dataset,
    block_size: int,
    chunk_size: int,
) -> Recommender:
    """The model to evaluate: restored from its checkpoint when one is given.

    A closed-form model keeps what it learned in plain attributes, which the
    checkpoint carries, so it is restored from them rather than refitted. An
    iterative model is built from the configuration and given the saved weights.

    Args:
        model_name (str): The name of the model.
        model_params (Dict[str, Any]): Its configured parameters.
        dataset (Dataset): The dataset the model is evaluated on.
        block_size (int): The block size of the model.
        chunk_size (int): The chunk size of the model.

    Returns:
        Recommender: The model, ready to be evaluated.
    """
    params = model_param_from_dict(model_name, model_params)
    load_from = params.meta.load_from
    model_class = model_registry.get_class(model_name)

    checkpoint = None
    if load_from is not None:
        checkpoint = torch.load(load_from, weights_only=False, map_location="cpu")
        _check_checkpoint_matches(checkpoint, model_class, dataset, load_from)

        if not issubclass(model_class, IterativeRecommender):
            model = model_class.from_checkpoint(checkpoint=checkpoint)
            logger.positive(f"Restored {model_name} from {load_from}.")
            return model

    model = model_registry.get(
        name=model_name,
        params=model_params,
        interactions=dataset.train_set,
        transactions=dataset.train_transactions,
        knowledge=dataset.knowledge,
        multimodal=dataset.multimodal,
        sessions=dataset.train_session,
        seed=42,
        info=dataset.info(),
        **dataset.get_stash(),
        block_size=block_size,
        chunk_size=chunk_size,
    )

    if isinstance(model, IterativeRecommender):
        if checkpoint is not None:
            model.load_state_dict(checkpoint["state_dict"])
            logger.positive(f"Loaded the weights of {model_name} from {load_from}.")
        else:
            logger.negative(
                "No checkpoint path found. Model will be evaluated using default weights."
            )
    return model


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

    # A missing checkpoint is found before any data is read
    _check_checkpoint_paths(config.models)

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
    # One re-ranker for both paths, so the list reported is the list written.
    reranker = build_reranker(config.rerank, main_dataset)
    evaluator = build_evaluator(config.evaluation, main_dataset, reranker)

    # Experiment device
    general_device = config.general.device

    data_preparation_time = time.time() - experiment_start_time
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )

    for model_name, model_params in config.models.items():
        params = model_param_from_dict(model_name, model_params)

        # Evaluation params
        block_size = params.optimization.block_size
        chunk_size = params.optimization.chunk_size
        num_workers = resolve_num_workers(
            params.optimization.num_workers,
            resolve_available_cpus(params.optimization.cpu_per_trial),
        )

        # Model device
        model_device = params.optimization.device
        device = general_device if model_device is None else model_device
        evaluation_dataloader_kwargs = build_evaluation_dataloader_kwargs(
            num_workers=num_workers,
            device=device,
            reuse_loader=False,
        )

        model = _load_or_build_model(
            model_name=model_name,
            model_params=model_params,
            dataset=main_dataset,
            block_size=block_size,
            chunk_size=chunk_size,
        )

        # Callback on training complete
        callback.on_training_complete(model=model)

        # Retrieve appropriate evaluation dataloader
        dataloader = retrieve_evaluation_dataloader(
            dataset=main_dataset,
            model=model,
            strategy=config.evaluation.strategy,
            num_negatives=config.evaluation.num_negatives,
            negative_sampling=config.evaluation.negative_sampling,
            neg_alpha=config.evaluation.neg_alpha,
            **evaluation_dataloader_kwargs,
        )
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
        log_evaluation(results, "Test", config.evaluation.max_metric_per_row)

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
                reranker=reranker,
                mask_seen=config.evaluation.mask_seen,
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
                        model_results,
                        stat_name,
                        backend=config.general.backend,
                        **corrections,
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
