import argparse
from argparse import Namespace

from elliotwo.data import LocalReader, LocalWriter, Splitter
from elliotwo.utils.config import load_yaml
from elliotwo.utils.ray_utils import parse_params
from elliotwo.utils.logger import logger
from elliotwo.recommenders.trainer import Trainer
from elliotwo.evaluation.evaluator import Evaluator
from elliotwo.evaluation.metrics import AbstractMetric
from elliotwo.utils.dataclasses import TrainerConfig
from elliotwo.utils.registry import metric_registry


def main(args: Namespace):
    """Main function to start the experiment.

    This method will start the train pipeline.
    """
    logger.msg("Starting experiment.")

    # Config parser testing
    config = load_yaml(args.config)

    # Writer module testing
    writer = LocalWriter(config)

    # Reader module testing
    reader = LocalReader(config)

    # Dataset loading
    dataset = None
    if config.data.loading_strategy == "dataset":
        data = reader.read()

        # Splitter testing
        if config.splitter:
            splitter = Splitter(config)

            if config.data.data_type == "transaction":
                dataset = splitter.split_transaction(data)
            else:
                raise ValueError("Data type not yet supported.")
        else:
            # This branch is for 100% train and 0% test
            pass

    elif config.data.loading_strategy == "split":
        if config.data.data_type == "transaction":
            dataset = reader.read_transaction_split()
        else:
            raise ValueError("Data type not yet supported.")

    if config.splitter and config.splitter.save_split:
        writer.write_split(dataset)

    # Trainer testing
    models = list(config.models.keys())
    coo = Evaluator(dataset, config)

    for model_name in models:
        params = config.models[model_name]
        val_metric, val_k = config.validation_metric(
            params["optimization"]["validation_metric"]
        )
        metric: AbstractMetric = metric_registry.get(val_metric, config=config)
        train_params = parse_params(params)
        train_config = TrainerConfig(model_name, train_params, metric, val_k)
        trainer = Trainer(dataset, train_config, config)
        best_model, _ = trainer.train_and_evaluate()

        # Evaluation testing
        result_dict = coo.run(best_model)
        writer.write_results(
            result_dict, model_name, config.evaluation.metrics, config.evaluation.top_k
        )

        # Recommendation
        if config.general.recommendation.save_recs:
            umap_i, imap_i = dataset.get_inverse_mappings()
            recs = best_model.get_recs(umap_i, imap_i, k=50)
            writer.write_recs(recs, model_name)

        # Model serialization
        if params["meta"]["save_model"]:
            best_model.save_model(writer)


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
