from typing import Optional, ClassVar, Dict, Callable, Any

from pydantic import BaseModel, field_validator

from warprec.utils.config.common import validate_callback


class CustomCallbacksConfig(BaseModel):
    """Definition of the custom callbacks configuration part of the configuration file.

    This class reads custom scripts, loading callbacks to use during training.

    Attributes:
        on_model_evaluation (Optional[str]): Path to the script containing the callback
            to run after each model evaluation.
        on_dataset_creation (Optional[str]): Path to the script containing the callback
            to run after dataset creation.

    Raises:
        ValueError: If the callback script cannot be loaded or the function is not found.
    """

    on_model_evaluation: Optional[str] = None
    on_dataset_creation: Optional[str] = None

    # Callback loaded will be stored directly inside the Pydantic model
    _loaded_callbacks: ClassVar[
        Dict[str, Callable[[Any, Dict[str, Any], Any], None]]
    ] = {}

    @field_validator("on_model_evaluation", mode="before")
    @classmethod
    def check_on_model_evaluation(cls, v: str):
        if v is None:
            return

        loaded_func = validate_callback(v, "on_model_evaluation")

        if loaded_func:
            # If found, store the loaded function in the class variable
            cls._loaded_callbacks["on_model_evaluation"] = loaded_func
        else:
            raise ValueError(
                f"Failed to load 'on_model_evaluation' callback from '{v}'. See logs for details."
            )

        return v

    @field_validator("on_dataset_creation", mode="before")
    @classmethod
    def check_on_dataset_creation(cls, v: str):
        if v is None:
            return

        loaded_func = validate_callback(v, "on_dataset_creation")

        if loaded_func:
            # If found, store the loaded function in the class variable
            cls._loaded_callbacks["on_dataset_creation"] = loaded_func
        else:
            raise ValueError(
                f"Failed to load 'on_dataset_creation' callback from '{v}'. See logs for details."
            )

        return v
