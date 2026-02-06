# pylint: disable=too-many-branches, too-many-statements
from typing import Tuple, List, Optional, Dict, Union, Any

from narwhals.dataframe import DataFrame

from warprec.data import Dataset
from warprec.data.reader import Reader
from warprec.data.splitting import Splitter
from warprec.data.filtering import apply_filtering
from warprec.recommenders.base_recommender import ContextRecommenderUtils
from warprec.utils.config import (
    TrainConfiguration,
    DesignConfiguration,
    EvalConfiguration,
)
from warprec.utils.callback import WarpRecCallback
from warprec.utils.registry import model_registry
from warprec.utils.logger import logger


def initialize_datasets(
    reader: Reader,
    callback: WarpRecCallback,
    config: Union[TrainConfiguration, DesignConfiguration, EvalConfiguration],
) -> Tuple[Dataset, Dataset | None, List[Dataset]]:
    """Initialize datasets based on the configuration. This is a common operation
    used in both training and design scripts.

    Args:
        reader (Reader): The initialized reader object that will be used to read data.
        callback (WarpRecCallback): The callback object for handling events during initialization.
        config (Union[TrainConfiguration, DesignConfiguration, EvalConfiguration]): The configuration
            object containing all necessary settings for data loading, filtering, and splitting.

    Returns:
        Tuple[Dataset, Dataset | None, List[Dataset]]: A tuple containing the main
            dataset, an optional validation dataset, and a list of datasets for cross-validation folds.

    Raises:
        ValueError: If the data type specified in the configuration is not supported.
    """
    # Dataset loading
    main_dataset: Dataset = None
    val_data: List[Tuple[DataFrame[Any], DataFrame[Any]]] | DataFrame[Any] = None
    train_data: DataFrame[Any] = None
    test_data: DataFrame[Any] = None
    side_data = None
    user_cluster = None
    item_cluster = None
    if config.reader.loading_strategy == "dataset":
        file_format = config.reader.file_format

        match file_format:
            case "tabular":
                data = reader.read_tabular(
                    **config.reader.model_dump(exclude=["labels", "dtypes"]),  # type: ignore[arg-type]
                    column_names=config.reader.column_names(),
                    dtypes=config.reader.column_dtype(),
                )
            case "parquet":
                data = reader.read_parquet(
                    **config.reader.model_dump(exclude=["labels", "dtypes"]),  # type: ignore[arg-type]
                )
            case _:
                raise ValueError(f"File format '{file_format}'not supported.")
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
                    **config.reader.labels.model_dump(
                        exclude=["cluster_label", "context_labels"]  # type: ignore[arg-type]
                    ),
                    **test_configuration,
                    **val_configuration,
                )

            else:
                raise ValueError("Data type not yet supported.")

    elif config.reader.loading_strategy == "split":
        if config.reader.data_type == "transaction":
            file_format = config.reader.split.file_format

            match file_format:
                case "tabular":
                    train_data, val_data, test_data = reader.read_tabular_split(
                        **config.reader.split.model_dump(),
                        column_names=config.reader.column_names(),
                        dtypes=config.reader.column_dtype(),
                    )
                case "parquet":
                    train_data, val_data, test_data = reader.read_parquet_split(
                        **config.reader.split.model_dump(),
                        column_names=config.reader.column_names(),
                    )
                case _:
                    raise ValueError(f"File format '{file_format}'not supported.")
        else:
            raise ValueError("Data type not yet supported.")

    # Side information reading
    if config.reader.side:
        file_format = config.reader.split.file_format

        match file_format:
            case "tabular":
                side_data = reader.read_tabular(
                    **config.reader.side.model_dump(),
                )
            case "parquet":
                side_data = reader.read_parquet(
                    **config.reader.side.model_dump(),
                )
            case _:
                raise ValueError(f"File format '{file_format}'not supported.")

    # Cluster information reading
    if config.reader.clustering:

        def _read_cluster_data_clean(
            specific_config: dict,
            common_cluster_label: str,
            common_cluster_type: str,
            reader: Reader,
        ) -> DataFrame[Any]:
            """Reads clustering data using a pre-prepared specific configuration (User or Item).

            Args:
                specific_config (dict): Specific configurations for user or item.
                common_cluster_label (str): Common label for the cluster column.
                common_cluster_type (str): Common data type for the cluster column.
                reader (Reader): Object or module with the read_tabular method.

            Returns:
                DataFrame[Any]: A DataFrame containing the cluster data.

            Raises:
                ValueError: If the file format is not supported.
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

            # Read data using the custom reader
            file_format = specific_config["file_format"]

            match file_format:
                case "tabular":
                    cluster_data = reader.read_tabular(
                        local_path=specific_config["local_path"],
                        blob_name=specific_config["blob_name"],
                        column_names=column_names,
                        dtypes=dtype_map,
                        sep=specific_config["sep"],
                        header=specific_config["header"],
                    )

                case "parquet":
                    cluster_data = reader.read_parquet(
                        local_path=specific_config["local_path"],
                        blob_name=specific_config["blob_name"],
                        column_names=column_names,
                    )

                case _:
                    raise ValueError(f"File format '{file_format}'not supported.")

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
            "file_format": config.reader.clustering.user_file_format,
            "sep": config.reader.clustering.user_sep,
            "header": config.reader.clustering.user_header,
        }

        # Item specific clustering information
        item_config = {
            "id_label": config.reader.labels.item_id_label,
            "id_type": config.reader.dtypes.item_id_type,
            "local_path": config.reader.clustering.item_local_path,
            "blob_name": config.reader.clustering.item_azure_blob_name,
            "file_format": config.reader.clustering.item_file_format,
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
    common_params: Dict[str, Any] = {
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
        "context_labels": config.reader.labels.context_labels,
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

    def prepare_evaluation_loaders(
        dataset: Dataset, has_classic: bool, has_context: bool
    ):
        """utility function to prepare the evaluation dataloaders
        for a given dataset based on the evaluation strategy.

        Args:
            dataset (Dataset): The dataset to prepare.
            has_classic (bool): Wether or not experiment has a classic recommender.
            has_context (bool): Wether or not experiment has a context recommender.

        Raises:
            ValueError: If both the flags are False.
        """
        if not has_classic and not has_context:
            raise ValueError(
                "Something went wrong. No correct model found during evaluation "
                "initialization."
            )
        strategy = config.evaluation.strategy

        # Initialize the classic evaluation structures
        if has_classic:
            if strategy == "full":
                dataset.get_evaluation_dataloader()
            elif strategy == "sampled":
                dataset.get_sampled_evaluation_dataloader(
                    num_negatives=config.evaluation.num_negatives,
                    seed=config.evaluation.seed,
                )

        # Initialize the contextual evaluation structures
        if has_context:
            if strategy == "full":
                dataset.get_contextual_evaluation_dataloader()
            elif strategy == "sampled":
                dataset.get_sampled_contextual_evaluation_dataloader(
                    num_negatives=config.evaluation.num_negatives,
                    seed=config.evaluation.seed,
                )

    logger.msg("Preparing main dataset inner structures for evaluation.")

    model_classes = [
        model_registry.get_class(model_name) for model_name in config.models.keys()
    ]
    has_classic = any(
        not issubclass(model_class, ContextRecommenderUtils)
        for model_class in model_classes
    )
    has_context = any(
        issubclass(model_class, ContextRecommenderUtils)
        for model_class in model_classes
    )

    prepare_evaluation_loaders(main_dataset, has_classic, has_context)
    if fold_dataset is not None and isinstance(fold_dataset, list):
        for i, dataset in enumerate(fold_dataset):
            logger.msg(
                f"Preparing fold dataset {i + 1}/{len(fold_dataset)} inner structures for evaluation."
            )
            prepare_evaluation_loaders(dataset, has_classic, has_context)
