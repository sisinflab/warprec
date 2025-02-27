import argparse
from argparse import Namespace
from os.path import join

import elliotwo
from elliotwo.utils.config import load_yaml
from elliotwo.utils.logger import logger
from elliotwo.evaluation.evaluator import Evaluator
from elliotwo.utils.registry import model_registry


def main(args: Namespace):
    """Main function to start the experiment.

    This method will start the inference pipeline.
    """
    logger.msg("Starting inference.")

    # Config parser testing
    config = load_yaml(args.config)

    # Reader module testing
    reader = elliotwo.data.LocalReader(config)

    # Dataset loading
    dataset = None
    if config.data.loading_strategy == "dataset":
        data = reader.read()

        # Splitter testing
        if config.splitter:
            splitter = elliotwo.Splitter(config)

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

    # Models to infer
    models = list(config.models.keys())
    coo = Evaluator(dataset, config)

    for model_name in models:
        # Restoring model to previous state
        params = config.models[model_name]
        path = params["meta"]["load_from"]
        path = join(path, "serialized", model_name + ".joblib")
        deserialized_data = reader.load_model_state(local_path=path)

        imp = params["meta"]["implementation"]
        if imp == "latest":
            model = model_registry.get_latest(
                model_name, config=config, dataset=dataset, params=None
            )
        else:
            model = model_registry.get(
                model_name,
                implementation=imp,
                config=config,
                dataset=dataset,
                params=None,
            )
        model.load_model(deserialized_data)
        _ = coo.run(model)


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
