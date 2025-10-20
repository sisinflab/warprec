import argparse
import time
from argparse import Namespace

from pandas import DataFrame

from warprec.data.reader import ReaderFactory, Reader
from warprec.data.splitting import Splitter
from warprec.data.dataset import Dataset
from warprec.data.filtering import apply_filtering
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

    # Dataset loading
    main_dataset: Dataset = None
    train_data: DataFrame = None
    test_data: DataFrame = None
    side_data = None
    user_cluster = None
    item_cluster = None
    if config.reader.loading_strategy == "dataset":
        data = reader.read_tabular(
            local_path=config.reader.local_path,
            blob_name=config.reader.azure_blob_name,
            column_names=config.reader.column_names(),
            dtypes=config.reader.column_dtype(),
            sep=config.reader.sep,
            header=config.reader.header,
        )
        data = callback.on_data_reading(data)

        # Check for optional filtering
        if config.filtering is not None:
            filters = config.get_filters()
            data = apply_filtering(data, filters)

        # Splitter testing
        if config.splitter:
            splitter = Splitter(config)

            if config.reader.data_type == "transaction":
                train_data, _, test_data = splitter.split_transaction(data)

            else:
                raise ValueError("Data type not yet supported.")

    elif config.reader.loading_strategy == "split":
        if config.reader.data_type == "transaction":
            train_data, _, test_data = reader.read_tabular_split(
                split_dir=config.reader.split.local_path,
                blob_prefix=config.reader.split.azure_blob_prefix,
                column_names=config.reader.column_names(),
                dtypes=config.reader.column_dtype(),
                sep=config.reader.split.sep,
                ext=config.reader.split.ext,
                header=config.reader.split.header,
            )

        else:
            raise ValueError("Data type not yet supported.")

    # Side information reading
    if config.reader.side:
        side_data = reader.read_tabular(
            local_path=config.reader.side.local_path,
            blob_name=config.reader.side.azure_blob_name,
            sep=config.reader.side.sep,
            header=config.reader.side.header,
        )

    # Cluster information reading
    if config.reader.clustering:

        def _read_cluster_data_clean(
            specific_config: dict,
            common_cluster_label: str,
            common_cluster_type: str,
            reader: Reader,
        ) -> DataFrame:
            """Reads clustering data using a pre-prepared specific configuration (User or Item).

            Args:
                specific_config (dict): Specific configurations for user or item.
                common_cluster_label (str): Common label for the cluster column.
                common_cluster_type (str): Common data type for the cluster column.
                reader (Reader): Object or module with the read_tabular method.

            Returns:
                DataFrame: A Pandas DataFrame containing the cluster data.
            """

            # Define column names
            column_names = [
                specific_config["id_label"],
                common_cluster_label,
            ]

            # Define data types (and map them to column names)
            dtypes_list = [
                specific_config["id_type"],
                common_cluster_type,
            ]
            dtype_map = zip(column_names, dtypes_list)

            # Read tabular data using the custom reader
            cluster_data = reader.read_tabular(
                local_path=specific_config["local_path"],
                blob_name=specific_config["blob_name"],
                column_names=column_names,
                dtypes=dtype_map,
                sep=specific_config["sep"],
                header=specific_config["header"],
            )

            return cluster_data

        # Common clustering information
        common_cluster_label = config.reader.labels.cluster_label
        common_cluster_type = config.reader.dtypes.cluster_type

        # User specific clustering information
        user_config = {
            "id_label": config.reader.labels.user_id_label,
            "id_type": config.reader.dtypes.user_id_type,
            "local_path": config.reader.clustering.user_local_path,
            "blob_name": config.reader.clustering.user_azure_blob_name,
            "sep": config.reader.clustering.user_sep,
            "header": config.reader.clustering.user_header,
        }

        # Item specific clustering information
        item_config = {
            "id_label": config.reader.labels.item_id_label,
            "id_type": config.reader.dtypes.item_id_type,
            "local_path": config.reader.clustering.item_local_path,
            "blob_name": config.reader.clustering.item_azure_blob_name,
            "sep": config.reader.clustering.item_sep,
            "header": config.reader.clustering.item_header,
        }

        # Read user clustering data
        user_cluster = _read_cluster_data_clean(
            specific_config=user_config,
            common_cluster_label=common_cluster_label,
            common_cluster_type=common_cluster_type,
            reader=reader,
        )

        # Read item clustering data
        item_cluster = _read_cluster_data_clean(
            specific_config=item_config,
            common_cluster_label=common_cluster_label,
            common_cluster_type=common_cluster_type,
            reader=reader,
        )

    # Dataset common information
    common_params = {
        "side_data": side_data,
        "user_cluster": user_cluster,
        "item_cluster": item_cluster,
        "batch_size": config.evaluation.batch_size,
        "rating_type": config.reader.rating_type,
        "rating_label": config.reader.labels.rating_label,
        "timestamp_label": config.reader.labels.timestamp_label,
        "cluster_label": config.reader.labels.cluster_label,
        "precision": config.general.precision,
    }

    # Create main dataset
    logger.msg("Creating main dataset")
    main_dataset = Dataset(
        train_data,
        test_data,
        **common_params,
    )

    # Callback on dataset creation
    callback.on_dataset_creation(
        main_dataset=main_dataset,
        val_dataset=None,
        validation_folds=[],
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
