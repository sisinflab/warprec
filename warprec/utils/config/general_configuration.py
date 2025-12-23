import os
import sys
import importlib
from pathlib import Path
from typing import Optional, Type, List, Dict, Any

import torch
from pydantic import BaseModel, field_validator, model_validator, Field

from warprec.utils.helpers import is_python_module
from warprec.utils.callback import WarpRecCallback


class WarpRecCallbackConfig(BaseModel):
    """Definition of the custom callback configuration part of the configuration file.

    This class reads a python script and load the custom implemented callback.

    Attributes:
        callback_path (Optional[str]): Path to the script containing the callback.
        callback_name (Optional[str]): Name of the callback to load from the script.
        args (Optional[List[Any]]): Positional arguments to pass to the callback.
        kwargs (Optional[Dict[str, Any]]): Additional keyword arguments to pass to the callback.

    Note:
        The args dictionary can be used to pass additional parameters but
        it is not validated. It is the user's responsibility to ensure that
        the parameters passed in args are correct.

    Raises:
        ValueError: If the callback script cannot be found or is incorrect.
    """

    callback_path: Optional[str] = None
    callback_name: Optional[str] = None
    args: Optional[List[Any]] = []
    kwargs: Optional[Dict[str, Any]] = {}

    @field_validator("callback_path")
    @classmethod
    def check_callback_path(cls, v: str):
        if v is None:
            return

        if not isinstance(v, str):
            raise ValueError(
                f"Callback script path must be a string, got: {type(v).__name__}."
            )

        if not v.endswith(".py"):
            raise ValueError(
                f"Callback script path must be a Python file ('.py' extension), got: {v}."
            )

        if not os.path.exists(v):
            raise ValueError(f"Callback script file not found at path: {v}.")

        return v

    @model_validator(mode="after")
    def validate_callback_class_and_path(self) -> "WarpRecCallbackConfig":
        """Validates that if callback_path is provided, callback_name is also provided,
        and that the specified class exists and is a subclass of WarpRecCallback."""
        if self.callback_path is None and self.callback_name is None:
            # No callback provided by the user, so we can skip validation
            return self

        if self.callback_path is None and self.callback_name is not None:
            raise ValueError("callback_name cannot be provided without callback_path.")

        if self.callback_path is not None and self.callback_name is None:
            raise ValueError("callback_path cannot be provided without callback_name.")

        # Here we know that both field are provided, so we need
        # to check if the implementation exists
        try:
            # Try to load the custom callback class
            self._load_and_validate_custom_callback_class(
                self.callback_path, self.callback_name
            )
            return self
        except Exception as e:
            raise ValueError(f"Error validating custom callback: {e}")

    def _load_and_validate_custom_callback_class(
        self, module_path: str, class_name: str
    ) -> Type[WarpRecCallback]:
        """Internal helper to load the module and validate the class."""
        mod_path = Path(module_path)
        module_dir = mod_path.parent
        original_sys_path = sys.path[:]

        try:
            if str(module_dir) not in sys.path:
                sys.path.insert(0, str(module_dir))

            spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
            if spec is None:
                raise ImportError(f"Could not load spec for module: {module_path}.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, class_name):
                raise ValueError(
                    f"Class '{class_name}' not found in module: {module_path}."
                )

            loaded_class = getattr(module, class_name)

            # Check if the specified class is an implementation of
            # WarpRec custom callback
            if not issubclass(loaded_class, WarpRecCallback):
                raise ValueError(
                    f"Class '{class_name}' must inherit from 'WarpRecCallback'."
                )

            return loaded_class

        finally:
            # Restore the original sys.path
            sys.path = original_sys_path


class AzureConfig(BaseModel):
    """Configuration for Azure services.

    Attributes:
        storage_account_name (str): The name of the Azure Storage Account.
        container_name (str): The name of the Azure Container.
    """

    storage_account_name: str
    container_name: str


class GeneralConfig(BaseModel):
    """Definition of the general configuration part of the configuration file.

    This class reads all the side information about the experiment from the configuration file.

    Attributes:
        precision (Optional[str]): The precision to use during computation.
        device (Optional[str]): The device that will be used for tensor operations.
        ray_verbose (Optional[int]): The Ray level of verbosity.
        time_report (Optional[bool]): Whether to report the time taken by each step.
        train_data_preparation (Optional[str]): Defines how to prepare training data structures.
            Can either be:
            - None: No preparation will be executed.
            - "conservative": Data structures will be prepared only for the current model, then
                the cache will be cleared.
            - "experiment": Data structures will be prepared for the whole experiment.
                Will require more memory.
        custom_models (Optional[str | List[str]]): List of custom model paths to load.
            This is useful for loading custom models that are not part of the
            standard Warprec framework.
        callback (Optional[WarpRecCallbackConfig]): The custom callback configuration.
        azure (Optional[AzureConfig]): The Azure configuration.
    """

    precision: Optional[str] = "float32"
    device: Optional[str] = "cpu"
    ray_verbose: Optional[int] = 1
    time_report: Optional[bool] = True
    train_data_preparation: Optional[str] = None
    custom_models: Optional[str | List[str]] = None
    callback: Optional[WarpRecCallbackConfig] = Field(
        default_factory=WarpRecCallbackConfig
    )
    azure: Optional[AzureConfig] = None

    @field_validator("device")
    @classmethod
    def check_device(cls, v: str):
        """Validate device."""
        if v in ("cuda", "cpu"):
            if v == "cuda" and not torch.cuda.is_available():
                raise ValueError(
                    "Cuda device was selected but not available on current machine."
                )
            return v
        raise ValueError(f'Device {v} is not supported. Use "cpu" or "cuda".')

    @field_validator("train_data_preparation")
    @classmethod
    def check_train_data_preparation(cls, v: str):
        """Validate train_data_preparation."""
        if v is None:
            return v
        if v in ["conservative", "experiment"]:
            return v
        raise ValueError(
            f"Train data preparation value can either be: None, 'conservative' or 'experiment'. "
            f"Value received: {v}"
        )

    @field_validator("custom_models")
    @classmethod
    def check_custom_models(cls, v: None | str | List[str]) -> List[str]:
        """Validates the custom models list."""
        if v is None:
            return []

        if isinstance(v, str):
            if is_python_module(v):
                return [v]
            raise ValueError(f"Custom model path '{v}' is not a valid Python module.")

        for path in v:
            if not isinstance(path, str):
                raise ValueError(
                    f"Each custom model path must be a string, got: {path}."
                )
            if not is_python_module(path):
                raise ValueError(
                    f"Custom model path '{path}' is not a valid Python module."
                )
        return v
