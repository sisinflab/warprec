import importlib
from typing import List, Tuple, TYPE_CHECKING
from pathlib import Path

from warprec.utils.config.model_configuration import RecomModel
from warprec.utils.registry import params_registry
from warprec.utils.logger import logger

if TYPE_CHECKING:
    from warprec.data.dataset import (
        Dataset,
        EvaluationDataLoader,
        NegativeEvaluationDataLoader,
    )


def load_custom_modules(custom_modules: str | List[str] | None):
    """Load custom modules dynamically.

    Args:
        custom_modules (str | List[str] | None): List of custom module paths to import.
            Each path can either point to a Python file (ending with .py)
            or a directory containing an __init__.py file.
    """
    if isinstance(custom_modules, str):
        custom_modules = [custom_modules]

    # Import custom models dynamically
    if custom_modules is not None and len(custom_modules) > 0:
        for model_path in custom_modules:
            model_path = model_path.removesuffix(".py")
            try:
                importlib.import_module(model_path)
            except ImportError as e:
                logger.negative(f"Failed to import custom model {model_path}: {e}")


def is_python_module(path: str | Path) -> bool:
    """Check if the given path is a valid Python module.

    Args:
        path (str | Path): The path to check.

    Returns:
        bool: True if the path is a valid Python module, False otherwise.
    """
    path = Path(path)

    if path.is_file() and path.suffix == ".py":
        return True

    if path.is_dir() and (path / "__init__.py").is_file():
        return True

    return False


def validation_metric(val_metric: str) -> Tuple[str, int]:
    """This method will parse the validation metric.

    Args:
        val_metric (str): The validation metric in string format.

    Returns:
        Tuple[str, int]:
            str: The name of the metric to use for validation.
            int: The cutoff to use for validation.
    """
    metric_name, top_k = val_metric.split("@")
    return metric_name, int(top_k)


def model_param_from_dict(model_name: str, params: dict) -> "RecomModel":
    """Retrieve the Pydantic model and validate the parameters.

    Args:
        model_name (str): The name of the model.
        params (dict): The parameter dictionary.

    Returns:
        RecomModel: The validated parameter model.
    """
    model_params: RecomModel = (
        params_registry.get(model_name, **params)
        if model_name.upper() in params_registry.list_registered()
        else RecomModel(**params)
    )
    return model_params


def retrieve_evaluation_dataloader(
    dataset: "Dataset",
    strategy: str,
    num_negatives: int = 99,
    seed: int = 42,
) -> "EvaluationDataLoader" | "NegativeEvaluationDataLoader":
    """Retrieve the appropriate evaluation dataloader based on the strategy.

    Args:
        dataset (Dataset): The dataset containing train, val, and test sets.
        strategy (str): The evaluation strategy ('full' or 'sampled').
        num_negatives (int): The number of negative samples per positive instance.
        seed (int): Random seed for negative sampling.

    Returns:
        EvaluationDataLoader | NegativeEvaluationDataLoader: The appropriate evaluation dataloader.

    Raises:
        ValueError: If an unknown evaluation strategy is provided.
    """
    dataloader: "EvaluationDataLoader" | "NegativeEvaluationDataLoader"
    if strategy == "full":
        dataloader = dataset.get_evaluation_dataloader()
    elif strategy == "sampled":
        dataloader = dataset.get_neg_evaluation_dataloader(
            num_negatives=num_negatives,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown evaluation strategy: {strategy}")

    return dataloader
