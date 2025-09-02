import importlib
from typing import List, Tuple
from pathlib import Path

from warprec.utils.logger import logger


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
