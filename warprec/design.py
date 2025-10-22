import argparse
import time
from argparse import Namespace

from warprec.common import initialize_datasets
from warprec.data.reader import ReaderFactory
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import load_design_configuration, load_callback
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry
from warprec.recommenders.loops import train_loop
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.evaluation.evaluator import Evaluator


def main(args: Namespace):
    """Main function to start the design pipeline.

    During the design execution you can test your custom models
    and debug them using a simpler version of the train pipeline.
    """
    logger.msg("Starting the Design Pipeline.")
    experiment_start_time = time.time()

    # Configuration loading
    config = load_design_configuration(args.config)

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
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
        compute_per_user=False,
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
        model = model_registry.get(
            name=model_name,
            params=params,
            interactions=main_dataset.train_set,
            device=device,
            seed=42,
            info=main_dataset.info(),
            **main_dataset.get_stash(),
        )

        if isinstance(model, IterativeRecommender):
            train_loop(model, main_dataset, model.epochs)

        # Callback on training complete
        callback.on_training_complete(model=model)

        # Evaluation on main dataset
        evaluator.evaluate(
            model,
            main_dataset,
            strategy=config.evaluation.strategy,
            num_negatives=config.evaluation.num_negatives,
            device=str(model._device),
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
