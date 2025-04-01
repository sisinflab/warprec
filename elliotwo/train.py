import argparse
import os
from argparse import Namespace

from elliotwo.data import LocalReader, LocalWriter, Splitter
from elliotwo.utils.config import load_yaml, parse_params
from elliotwo.utils.logger import logger
from elliotwo.recommenders.trainer import Trainer
from elliotwo.recommenders.abstract_recommender import generate_model_name
from elliotwo.evaluation.evaluator import Evaluator


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

    evaluator = Evaluator(
        list(config.evaluation.metrics),
        list(config.evaluation.top_k),
        train_set=dataset.train_set.get_sparse(),
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
    )

    for model_name in models:
        params = config.models[model_name]
        val_metric, val_k = config.validation_metric(
            params["optimization"]["validation_metric"]
        )
        train_params = parse_params(params)
        trainer = Trainer(model_name, train_params, dataset, val_metric, val_k, config)
        best_model, checkpoint_param = trainer.train_and_evaluate()

        # Evaluation testing
        result_dict = {}
        evaluator.evaluate(best_model, dataset, test_set=False, verbose=True)
        results = evaluator.compute_results()
        evaluator.print_console(results, config.evaluation.metrics, "Validation")
        result_dict["Validation"] = results

        evaluator.evaluate(best_model, dataset, test_set=True, verbose=True)
        results = evaluator.compute_results()
        evaluator.print_console(results, config.evaluation.metrics, "Test")
        result_dict["Test"] = results

        writer.write_results(
            result_dict, model_name, config.evaluation.metrics, config.evaluation.top_k
        )

        # Recommendation
        if config.general.recommendation.save_recs:
            umap_i, imap_i = dataset.get_inverse_mappings()
            recs = best_model.get_recs(
                dataset.train_set, umap_i, imap_i, k=config.general.recommendation.k
            )
            writer.write_recs(recs, model_name)

        # Model serialization
        if params["meta"]["save_model"]:
            writer.write_model(best_model)

            if params["meta"]["keep_all_ray_checkpoints"]:
                for check_path, param in checkpoint_param:
                    if os.path.exists(check_path):
                        source_path = os.path.join(check_path, "checkpoint.pt")
                        checkpoint_name = generate_model_name(model_name, param)
                        writer.checkpoint_from_ray(source_path, checkpoint_name)


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
