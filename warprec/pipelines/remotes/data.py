from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import ray

from warprec.errors import ConfigurationError
from warprec.recommenders.reranking import build_reranker
from warprec.common import (
    initialize_datasets,
    dataset_preparation,
)
from warprec.data.reader import Reader
from warprec.data.writer import Writer
from warprec.recommenders.base_recommender import Recommender
from warprec.data import Dataset
from warprec.data.timing import DataPreparationTimes
from warprec.utils.helpers import load_custom_modules
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import RecomModel, TrainConfiguration


@dataclass
class DataLayout:
    """What the driver knows of the datasets a data preparation task built.

    The datasets themselves stay in the object store, so this is all the driver
    fetches: how the data was split, and where the preparation spent its time.

    Attributes:
        has_validation (bool): Whether there is a validation split.
        n_folds (int): The number of cross-validation folds, 0 without them.
        times (DataPreparationTimes): The seconds the task spent reading,
            filtering, splitting and building each structure.
    """

    has_validation: bool = False
    n_folds: int = 0
    times: DataPreparationTimes = field(default_factory=DataPreparationTimes)


@ray.remote
def remote_data_preparation(
    reader: Reader,
    callback: WarpRecCallback,
    config: TrainConfiguration,
    writer: Optional[Writer] = None,
) -> Tuple[Dataset, Optional[Dataset], List[Dataset], DataLayout]:
    """Remote task to handle heavy data preparation on a worker node.

    This function offloads the memory-intensive operations (like joins, sorts,
    and DataFrame materialization) to a Ray worker, preventing Out-Of-Memory
    issues on the driver node. Called with 'num_returns=4', each dataset is an
    object of its own in the object store, and the driver fetches only the
    layout. The splits are written here, when asked for, for the same reason.

    Args:
        reader (Reader): The initialized reader object.
        callback (WarpRecCallback): The callback object.
        config (TrainConfiguration): The configuration object.
        writer (Optional[Writer]): The writer of the run, which writes the
            splits when 'writer.save_split' is set.

    Returns:
        Tuple[Dataset, Optional[Dataset], List[Dataset], DataLayout]: A
            tuple containing:
            - Dataset: The main dataset.
            - Optional[Dataset]: The validation dataset (if any).
            - List[Dataset]: The list of fold datasets for cross-validation.
            - DataLayout: How the data was split, and the seconds the task
              spent reading, filtering, splitting and building each structure,
              measured in the task, so without Ray's start-up and scheduling.
    """
    # Load custom modules if provided
    load_custom_modules(config.general.custom_modules)

    # Load datasets using common utility
    timings = DataPreparationTimes()
    main_dataset, val_dataset, fold_dataset = initialize_datasets(
        reader=reader,
        callback=callback,
        config=config,
        timings=timings,
    )

    # Prepare dataloaders for evaluation
    dataset_preparation(main_dataset, fold_dataset, config, timings=timings)

    if writer is not None:
        write_splits(writer, config, main_dataset, val_dataset, fold_dataset)

    layout = DataLayout(
        has_validation=val_dataset is not None,
        n_folds=len(fold_dataset or []),
        times=timings,
    )
    return main_dataset, val_dataset, fold_dataset, layout


def write_splits(
    writer: Writer,
    config: TrainConfiguration,
    main_dataset: Dataset,
    val_dataset: Optional[Dataset],
    fold_dataset: Optional[List[Dataset]],
) -> None:
    """Write the splits of the data, when the configuration asks for them.

    Args:
        writer (Writer): The writer of the run.
        config (TrainConfiguration): The configuration object.
        main_dataset (Dataset): The main dataset.
        val_dataset (Optional[Dataset]): The validation dataset (if any).
        fold_dataset (Optional[List[Dataset]]): The cross-validation folds.

    Raises:
        ConfigurationError: If the configured split file format is not supported.
    """
    if not (config.splitter and config.writer.save_split):
        return

    file_format = config.writer.split.file_format
    match file_format:
        case "tabular":
            writer.write_tabular_split(
                main_dataset,
                val_dataset,
                fold_dataset,
                **config.writer.split.model_dump(),
            )
        case "parquet":
            writer.write_parquet_split(
                main_dataset,
                val_dataset,
                fold_dataset,
                **config.writer.split.model_dump(),
            )
        case _:
            raise ConfigurationError(f"File format '{file_format}'not supported.")


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
    generate_recs(writer, model, dataset, config, device)
    return True


def generate_recs(
    writer: Writer,
    model: Recommender,
    dataset: Dataset,
    config: TrainConfiguration,
    device: str,
) -> None:
    """Write the recommendation lists of a model, in batches.

    Args:
        writer (Writer): The initialized writer object.
        model (Recommender): The trained model.
        dataset (Dataset): The dataset containing users/items.
        config (TrainConfiguration): The main configuration object.
        device (str): The device to use for inference.
    """
    # Move model to the correct device on the worker
    model.to(device)

    # Write recommendations in batches on shared file system
    writer.write_recs(
        reranker=build_reranker(config.rerank, dataset),
        mask_seen=config.evaluation.mask_seen,
        seed=config.evaluation.seed,
        model=model,
        dataset=dataset,
        **config.writer.recommendation.model_dump(),
    )


def write_model_artifacts(
    writer: Writer,
    config: TrainConfiguration,
    model_name: str,
    params: RecomModel,
    model: Recommender,
    results: Dict[Any, Any],
    dataset: Dataset,
) -> None:
    """Write what a run keeps of one evaluated model that needs its dataset.

    That is, where the model and the dataset are: the per-user results, which
    name the users by their original ids, and, when 'meta.save_model' is set,
    the model itself with what serving needs from the data.

    Args:
        writer (Writer): The writer of the run.
        config (TrainConfiguration): The configuration object.
        model_name (str): The name of the model.
        params (RecomModel): The parameters of the model.
        model (Recommender): The best model found.
        results (Dict[Any, Any]): The evaluation results of the model.
        dataset (Dataset): The main dataset the model was evaluated on.
    """
    if config.evaluation.save_per_user:
        i_umap, _ = dataset.get_inverse_mappings()
        writer.write_results_per_user(
            results, model_name, i_umap, **config.writer.results.model_dump()
        )

    # Model serialization, with what serving needs from the data
    if params.meta.save_model:
        # The seed travels with the model, so that a server breaks ties as the
        # evaluation and the written recommendations broke them.
        evaluation: Dict[str, Any] = {
            "strategy": config.evaluation.strategy,
            "seed": config.evaluation.seed,
        }
        if config.evaluation.strategy == "sampled":
            evaluation["num_negatives"] = config.evaluation.num_negatives
        writer.write_model(
            model,
            dataset=dataset,
            training={
                "dataset": config.writer.dataset_name,
                "evaluation": evaluation,
                "metrics": results,
            },
        )
