import argparse
from argparse import Namespace
from os.path import join

import torch
import warprec
from warprec.data.dataset import TransactionDataset
from warprec.utils.config import load_yaml
from warprec.utils.logger import logger
from warprec.evaluation.evaluator import Evaluator
from warprec.utils.registry import model_registry


def main(args: Namespace):
    """Main function to start the experiment.

    This method will start the inference pipeline.
    """
    logger.msg("Starting inference.")

    # Config parser testing
    config = load_yaml(args.config)

    # Reader module testing
    reader = warprec.data.LocalReader(config)

    # Dataset loading
    dataset = None
    if config.reader.loading_strategy == "dataset":
        data = reader.read()

        # Splitter testing
        if config.splitter:
            splitter = warprec.Splitter(config)

            if config.reader.data_type == "transaction":
                train, test, val = splitter.split_transaction(data)
                dataset = TransactionDataset(
                    train,
                    test,
                    val,
                    batch_size=1024,
                    rating_type=config.reader.rating_type,
                )
            else:
                raise ValueError("Data type not yet supported.")
        else:
            # This branch is for 100% train and 0% test
            pass

    elif config.reader.loading_strategy == "split":
        if config.reader.data_type == "transaction":
            train, test, val = reader.read_transaction_split()
            dataset = TransactionDataset(
                train, test, val, batch_size=1024, rating_type=config.reader.rating_type
            )
        else:
            raise ValueError("Data type not yet supported.")

    # Models to infer
    models = list(config.models.keys())

    evaluator = Evaluator(
        list(config.evaluation.metrics),
        list(config.evaluation.top_k),
        train_set=dataset.train_set.get_sparse(),
        beta=config.evaluation.beta,
        pop_ratio=config.evaluation.pop_ratio,
    )

    for model_name in models:
        # Restoring model to previous state
        params = config.models[model_name]
        path = params["meta"]["load_from"]
        path = join(path, "serialized", model_name + ".pt")
        checkpoint = torch.load(path)

        model = model_registry.get(
            model_name,
            implementation=params["meta"]["implementation"],
        )
        model.load_state_dict(checkpoint)

        evaluator.evaluate(model, dataset, test_set=False, verbose=True)
        results = evaluator.compute_results()
        evaluator.print_console(results, config.evaluation.metrics, "Validation")

        evaluator.evaluate(model, dataset, test_set=True, verbose=True)
        results = evaluator.compute_results()
        evaluator.print_console(results, config.evaluation.metrics, "Test")


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
