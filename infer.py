import importlib
from os.path import join

import elliotwo
from elliotwo.utils.config import load_yaml
from elliotwo.utils.logger import logger
from elliotwo.evaluation.evaluator import Evaluator


def main():
    """Main function to start the experiment.

    This method will start the inference pipeline.
    """
    logger.msg("Starting inference.")

    # Config parser testing
    config_path = "config/infer_config.yml"
    config = load_yaml(config_path)

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
    recommender_module = importlib.import_module("elliotwo.recommenders")
    coo = Evaluator(dataset, config)

    for model in models:
        # Restoring model to previous state
        params = config.models[model]
        train_params = config.convert_params(model, params)
        path = params["meta"]["load_from"]
        path = join(path, "serialized", model + ".joblib")
        deserialized_data = reader.load_model_state(local_path=path)

        model_class = getattr(recommender_module, model)
        loaded_model = model_class(config, dataset, train_params)
        loaded_model.load_model(deserialized_data)
        _ = coo.run(loaded_model)


if __name__ == "__main__":
    main()
