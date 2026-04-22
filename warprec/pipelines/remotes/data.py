from typing import List, Tuple, Optional

import ray

from warprec.common import (
    initialize_datasets,
    dataset_preparation,
)
from warprec.data.reader import Reader
from warprec.data.writer import Writer
from warprec.recommenders.base_recommender import Recommender
from warprec.data import Dataset
from warprec.utils.helpers import load_custom_modules
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import TrainConfiguration


@ray.remote
def remote_data_preparation(
    reader: Reader,
    callback: WarpRecCallback,
    config: TrainConfiguration,
) -> Tuple[Dataset, Optional[Dataset], List[Dataset]]:
    """Remote task to handle heavy data preparation on a worker node.

    This function offloads the memory-intensive operations (like joins, sorts,
    and DataFrame materialization) to a Ray worker, preventing Out-Of-Memory
    issues on the driver node.

    Args:
        reader (Reader): The initialized reader object.
        callback (WarpRecCallback): The callback object.
        config (TrainConfiguration): The configuration object.

    Returns:
        Tuple[Dataset, Optional[Dataset], List[Dataset]]: A tuple containing:
            - Dataset: The main dataset.
            - Optional[Dataset]: The validation dataset (if any).
            - List[Dataset]: The list of fold datasets for cross-validation.
    """
    # Load custom modules if provided
    load_custom_modules(config.general.custom_modules)

    # Load datasets using common utility
    main_dataset, val_dataset, fold_dataset = initialize_datasets(
        reader=reader,
        callback=callback,
        config=config,
    )

    # Prepare dataloaders for evaluation
    dataset_preparation(main_dataset, fold_dataset, config)

    return main_dataset, val_dataset, fold_dataset


@ray.remote
def remote_generate_recs(
    writer: Writer,
    model: Recommender,
    dataset: Dataset,
    config: TrainConfiguration,
    device: str,
) -> bool:
    """Remote task to handle inference for recommendations.

    Args:
        writer (Writer): The initialized writer object.
        model (Recommender): The trained model.
        dataset (Dataset): The dataset containing users/items.
        config (TrainConfiguration): The main configuration object.
        device (str): The device to use for inference.

    Returns:
        bool: True if successful.
    """
    # Load custom modules if provided
    load_custom_modules(config.general.custom_modules)

    # Move model to the correct device on the worker
    model.to(device)

    # Write recommendations in batches on shared file system
    writer.write_recs(
        model=model,
        dataset=dataset,
        **config.writer.recommendation.model_dump(),
    )

    return True
