import time
import os

import lightning as L

from warprec.common import initialize_datasets, log_evaluation
from warprec.data.reader import ReaderFactory
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.callbacks import WarpRecLightningIntegrationCallback
from warprec.evaluation.evaluator import Evaluator
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import load_design_configuration, load_callback
from warprec.utils.helpers import retrieve_evaluation_dataloader, model_param_from_dict
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry


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
        complex_metrics=config.evaluation.complex_metrics,
        feature_lookup=main_dataset.get_features_lookup(),
        user_cluster=main_dataset.get_user_cluster(),
        item_cluster=main_dataset.get_item_cluster(),
    )

    # Experiment device
    general_device = config.general.device

    data_preparation_time = time.time() - experiment_start_time
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )

    for model_name, model_params in config.models.items():
        params = model_param_from_dict(model_name, model_params)

        # Evaluation params
        seed = params.optimization.properties.seed
        block_size = params.optimization.block_size
        chunk_size = params.optimization.chunk_size
        num_workers = params.optimization.num_workers

        # Model device
        model_device = params.optimization.device
        device = general_device if model_device is None else model_device

        model = model_registry.get(
            name=model_name,
            params=model_params,
            interactions=main_dataset.train_set,
            seed=seed,
            info=main_dataset.info(),
            **main_dataset.get_stash(),
            block_size=block_size,
            chunk_size=chunk_size,
        )

        # Evaluation dataloader
        eval_dataloader = retrieve_evaluation_dataloader(
            dataset=main_dataset,
            model=model,
            strategy=config.evaluation.strategy,
            num_negatives=config.evaluation.num_negatives,
        )

        if isinstance(model, IterativeRecommender):
            # Set up the learning rate scheduler and the optimizer
            model.set_optimization_parameters(
                optimizer_config=params.optimization.optimizer,
                lr_scheduler_config=params.optimization.lr_scheduler,
            )

            # Dataloader settings
            if num_workers is None:
                available_cpus = os.cpu_count()
                num_workers = max(available_cpus - 1, 1)

            persistent_workers = num_workers > 0
            pin_memory = device == "cuda"

            # Train dataloader
            train_dataloader = model.get_dataloader(
                interactions=main_dataset.train_set,
                sessions=main_dataset.train_session,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )

            # Standard training loop
            integration_callback = WarpRecLightningIntegrationCallback(
                evaluator=evaluator,
                dataset=main_dataset,
                strategy=config.evaluation.strategy,
                early_stopping_config=params.early_stopping,
                validation_score=config.evaluation.validation_metric,
                mode=params.optimization.properties.mode,
            )
            trainer = L.Trainer(
                max_epochs=model.epochs,
                devices="auto",
                accelerator=device,
                num_sanity_val_steps=0,
                logger=False,
                enable_checkpointing=False,
                check_val_every_n_epoch=params.optimization.eval_every_n,
                callbacks=[integration_callback],
            )
            trainer.fit(
                model,
                train_dataloaders=train_dataloader,
                val_dataloaders=eval_dataloader,
            )

        # Callback on training complete
        callback.on_training_complete(model=model)

        # Evaluation on main dataset
        evaluator.evaluate(
            model=model,
            dataloader=eval_dataloader,
            strategy=config.evaluation.strategy,
            dataset=main_dataset,
            device=device,
            verbose=True,
        )
        results = evaluator.compute_results()
        log_evaluation(results, "Test", config.evaluation.max_metric_per_row)

        # Callback after complete evaluation
        callback.on_evaluation_complete(
            model=model,
            params=model_params,
            results=results,
        )

    logger.positive("Design pipeline executed successfully. WarpRec is shutting down.")
