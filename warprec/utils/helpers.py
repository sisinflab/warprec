import importlib
from typing import List, Optional

from warprec.utils.logger import logger


def load_custom_modules(custom_modules: Optional[List[str]]):
    """Load custom modules dynamically.

    Args:
        custom_modules (Optional[List[str]]): List of custom module paths to import.
            Each path can either point to a Python file (ending with .py)
            or a directory containing an __init__.py file.
    """
    # Import custom models dynamically
    if custom_modules is not None and len(custom_modules) > 0:
        for model_path in custom_modules:
            model_path = model_path.removesuffix(".py")
            try:
                importlib.import_module(model_path)
            except ImportError as e:
                logger.negative(f"Failed to import custom model {model_path}: {e}")
