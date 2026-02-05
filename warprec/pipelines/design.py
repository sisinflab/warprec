import time

from warprec.common import initialize_datasets
from warprec.data.reader import ReaderFactory
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import LRScheduler, load_design_configuration, load_callback
from warprec.utils.helpers import retrieve_evaluation_dataloader
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry
from warprec.recommenders.loops import train_loop
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.evaluation.evaluator import Evaluator


def design_pipeline(path: str):
    """Main function to start the design pipeline.

    During the design execution you can test your custom models
    and debug them using a simpler version of the train pipeline.

    Args:
        path (str): Path to the configuration file.
    """
    logger.msg("Starting the Design Pipeline.")
    experiment_start_time = time.time()

    # Configuration loading
    config = load_design_configuration(path)

    # Load custom callback if specified
    callback: WarpRecCallback = load_callback(
        config.general.callback,
        *config.general.callback.args,
        **config.general.callback.kwargs,
    )

    # Initialize I/O modules
    reader = ReaderFactory.get_reader(config=config)

    # Load datasets using common utility
    main_dataset, _, _ = initialize_datasets(
        reader=reader,
        callback=callback,
        config=config,
    )

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

    # Experiment device
    device = config.general.device

    data_preparation_time = time.time() - experiment_start_time
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )

    for model_name, params in config.models.items():
        # Evaluation params
        block_size = params.get("optimization", {}).get("block_size", 50)
        chunk_size = params.get("optimization", {}).get("chunk_size", 4096)

        model = model_registry.get(
            name=model_name,
            params=params,
            interactions=main_dataset.train_set,
            device=device,
            seed=42,
            info=main_dataset.info(),
            **main_dataset.get_stash(),
            block_size=block_size,
            chunk_size=chunk_size,
        )

        if isinstance(model, IterativeRecommender):
            lr_scheduler_params = params.get("optimization", {}).get(
                "lr_scheduler", None
            )
            lr_scheduler = (
                LRScheduler(**lr_scheduler_params)
                if lr_scheduler_params is not None
                else None
            )
            train_loop(model, main_dataset, model.epochs, lr_scheduler, device)

        # Callback on training complete
        callback.on_training_complete(model=model)

        # Retrieve appropriate evaluation dataloader
        dataloader = retrieve_evaluation_dataloader(
            dataset=main_dataset,
            model=model,
            strategy=config.evaluation.strategy,
            num_negatives=config.evaluation.num_negatives,
        )

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

        # Callback after complete evaluation
        callback.on_evaluation_complete(
            model=model,
            params=params,
            results=results,
        )

    logger.positive("Design pipeline executed successfully. WarpRec is shutting down.")
