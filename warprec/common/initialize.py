from typing import Tuple, List, Optional

from pandas import DataFrame
from itertools import product

from warprec.data.dataset import Dataset
from warprec.data.reader import Reader
from warprec.data.splitting import Splitter
from warprec.data.filtering import apply_filtering
from warprec.utils.config import TrainConfiguration, DesignConfiguration
from warprec.utils.callback import WarpRecCallback
from warprec.utils.enums import SearchSpace
from warprec.utils.registry import model_registry
from warprec.utils.logger import logger


def initialize_datasets(
    reader: Reader,
    callback: WarpRecCallback,
    config: TrainConfiguration | DesignConfiguration,
) -> Tuple[Dataset, Dataset | None, List[Dataset]]:
    """Initialize datasets based on the configuration. This is a common operation
    used in both training and design scripts.

    Args:
        reader (Reader): The initialized reader object that will be used to read data.
        callback (WarpRecCallback): The callback object for handling events during initialization.
        config (TrainConfiguration | DesignConfiguration): The configuration object containing
            all necessary settings for data loading, filtering, and splitting.

    Returns:
        Tuple[Dataset, Dataset | None, List[Dataset]]: A tuple containing the main
            dataset, an optional validation dataset, and a list of datasets for cross-validation folds.

    Raises:
        ValueError: If the data type specified in the configuration is not supported.
    """
    # Dataset loading
    main_dataset: Dataset = None
    val_data: List[Tuple[DataFrame, DataFrame]] | DataFrame = None
    train_data: DataFrame = None
    test_data: DataFrame = None
    side_data = None
    user_cluster = None
    item_cluster = None
    if config.reader.loading_strategy == "dataset":
        data = reader.read_tabular(
            **config.reader.model_dump(exclude=["labels", "dtypes"]),  # type: ignore[arg-type]
            column_names=config.reader.column_names(),
            dtypes=config.reader.column_dtype(),
        )
        data = callback.on_data_reading(data)

        # Check for optional filtering
        if config.filtering is not None:
            filters = config.get_filters()
            data = apply_filtering(data, filters)

        # Splitter testing
        if config.splitter:
            splitter = Splitter()

            if config.reader.data_type == "transaction":
                # Gather splitting configurations
                test_configuration = config.splitter.test_splitting.model_dump()
                val_configuration = config.splitter.validation_splitting.model_dump()

                # Add tag to distinguish test and validation keys
                test_configuration = {
                    f"test_{key}": value for key, value in test_configuration.items()
                }
                val_configuration = {
                    f"val_{key}": value for key, value in val_configuration.items()
                }

                # Compute splitting
                train_data, val_data, test_data = splitter.split_transaction(
                    data,
                    **config.reader.labels.model_dump(exclude=["cluster_label"]),  # type: ignore[arg-type]
                    **test_configuration,
                    **val_configuration,
                )

            else:
                raise ValueError("Data type not yet supported.")

    elif config.reader.loading_strategy == "split":
        if config.reader.data_type == "transaction":
            train_data, val_data, test_data = reader.read_tabular_split(
                **config.reader.split.model_dump(),
                column_names=config.reader.column_names(),
                dtypes=config.reader.column_dtype(),
            )

        else:
            raise ValueError("Data type not yet supported.")

    # Side information reading
    if config.reader.side:
        side_data = reader.read_tabular(
            **config.reader.side.model_dump(),
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
        "user_id_label": config.reader.labels.user_id_label,
        "item_id_label": config.reader.labels.item_id_label,
        "rating_label": config.reader.labels.rating_label,
        "timestamp_label": config.reader.labels.timestamp_label,
        "cluster_label": config.reader.labels.cluster_label,
        "precision": config.general.precision,
    }

    logger.msg("Creating main dataset")
    main_dataset = Dataset(
        train_data,
        test_data,
        **common_params,
    )

    # Handle validation data
    val_dataset: Dataset = None
    fold_dataset: List[Dataset] = []
    if val_data is not None:
        if not isinstance(val_data, list):
            # CASE 2: Train/Validation/Test
            logger.msg("Creating validation dataset")
            val_dataset = Dataset(
                train_data,
                val_data,
                evaluation_set="Validation",
                **common_params,
            )
        else:
            # CASE 3: Cross-Validation
            n_folds = len(val_data)
            for idx, fold in enumerate(val_data):
                logger.msg(f"Creating fold dataset {idx + 1}/{n_folds}")
                val_train, val_set = fold
                fold_dataset.append(
                    Dataset(
                        val_train,
                        val_set,
                        evaluation_set="Validation",
                        **common_params,
                    )
                )

    # Callback on dataset creation
    callback.on_dataset_creation(
        main_dataset=main_dataset,
        val_dataset=val_dataset,
        validation_folds=fold_dataset,
    )

    return main_dataset, val_dataset, fold_dataset


def dataset_preparation(
    main_dataset: Dataset,
    fold_dataset: Optional[List[Dataset]],
    config: TrainConfiguration,
):
    """This method prepares the dataloaders inside the dataset
    that will be passed to Ray during HPO. It is important to
    precompute these dataloaders before starting the optimization to
    avoid multiple computations of the same dataloader.

    Args:
        main_dataset (Dataset): The main dataset of train/test split.
        fold_dataset (Optional[List[Dataset]]): The list of validation datasets
            of train/val splits.
        config (TrainConfiguration): The configuration file used for the experiment.
    """

    def prepare_evaluation_loaders(dataset: Dataset):
        """utility function to prepare the evaluation dataloaders
        for a given dataset based on the evaluation strategy.

        Args:
            dataset (Dataset): The dataset to prepare.
        """
        if config.evaluation.strategy == "full":
            dataset.get_evaluation_dataloader()
        elif config.evaluation.strategy == "sampled":
            dataset.get_neg_evaluation_dataloader(
                num_negatives=config.evaluation.num_negatives,
                seed=config.evaluation.seed,
            )

    logger.msg("Preparing main dataset inner structures for evaluation.")

    prepare_evaluation_loaders(main_dataset)
    if fold_dataset is not None and isinstance(fold_dataset, list):
        for i, dataset in enumerate(fold_dataset):
            logger.msg(
                f"Preparing fold dataset {i + 1}/{len(fold_dataset)} inner structures for evaluation."
            )
            prepare_evaluation_loaders(dataset)

    def prepare_train_loaders(dataset: Dataset):
        models_configuration = config.models

        # Check each model requirements
        for model_name, params in models_configuration.items():
            model_class = model_registry.get_class(model_name)
            dataloader_requirement = model_class.DATALOADER_TYPE

            if dataloader_requirement is not None:
                loader_source = getattr(dataset, dataloader_requirement.source)
                loader_method = getattr(loader_source, dataloader_requirement.method)

                # Retrieve fixed parameters
                fixed_params = dataloader_requirement.fixed_params

                # Retrieve specific method parameter required
                construction_params = {}
                if len(dataloader_requirement.construction_params) > 0:
                    for con_param in dataloader_requirement.construction_params:
                        param_list = params[con_param]

                        # If the parameter is a grid or a choice, add it to the dict
                        if (
                            param_list[0] == SearchSpace.GRID
                            or param_list[0] == SearchSpace.CHOICE
                        ):
                            param_list = param_list[1:]

                        # If sampling from a specific search space, skip
                        if any(
                            param_list[0] == space
                            for space in [
                                SearchSpace.LOGRANDINT,
                                SearchSpace.LOGUNIFORM,
                                SearchSpace.QLOGRANDINT,
                                SearchSpace.QLOGUNIFORM,
                                SearchSpace.QRANDINT,
                                SearchSpace.QRANDN,
                                SearchSpace.QUNIFORM,
                                SearchSpace.RANDINT,
                                SearchSpace.RANDN,
                                SearchSpace.UNIFORM,
                            ]
                        ):
                            continue
                        construction_params[con_param] = param_list

                if construction_params:
                    # Create the product of all combinations
                    keys = list(construction_params.keys())
                    values = list(construction_params.values())

                    all_combination = product(*values)

                    # For each combination, call the constructor method
                    for combination in all_combination:
                        call_params = dict(zip(keys, combination))

                        # Try to call the method, ignore possible errors
                        try:
                            loader_method(**call_params, **fixed_params)
                        except Exception as e:
                            raise ValueError(
                                f"During dataset initialization a method failed with the following error: {e}"
                            )
                else:
                    # No specific parameters to pass to the method
                    loader_method(**fixed_params)

    logger.msg("Preparing main dataset inner structures for training.")

    prepare_train_loaders(main_dataset)
    if fold_dataset is not None and isinstance(fold_dataset, list):
        for i, dataset in enumerate(fold_dataset):
            logger.msg(
                f"Preparing fold dataset {i + 1}/{len(fold_dataset)} inner structures for training."
            )
            prepare_train_loaders(dataset)

    logger.positive("All dataset inner structures ready.")
