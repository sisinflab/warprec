import os
import sys
import importlib
from pathlib import Path
from typing import ClassVar, Dict, Any, List

import yaml
import numpy as np
import torch
from pydantic import BaseModel, model_validator, Field
from warprec.utils.helpers import load_custom_modules, validation_metric
from warprec.data.filtering import Filter
from warprec.utils.config import (
    GeneralConfig,
    WarpRecCallbackConfig,
    ReaderConfig,
    WriterConfig,
    SplittingConfig,
    DashboardConfig,
    RecomModel,
    EvaluationConfig,
)
from warprec.utils.callback import WarpRecCallback
from warprec.utils.enums import ReadingMethods
from warprec.utils.registry import model_registry, params_registry, filter_registry
from warprec.utils.logger import logger


class WarpRecConfiguration(BaseModel):
    """Definition of WarpRec base configuration file structure.

    Attributes:
        reader (ReaderConfig): Configuration of the reading process.
        filtering (Dict[str, dict]): The dictionary containing filtering
            information in the format {filter_name: dict{param_1: value, param_2: value, ...}, ...}
        models (Dict[str, dict]): The dictionary containing model information
            in the format {model_name: dict{param_1: value, param_2: value, ...}, ...}
        general (GeneralConfig): General configuration of the experiment
        sparse_np_dtype (ClassVar[dict]): The mapping between the string dtype
            and their numpy sparse counterpart.
        sparse_torch_dtype (ClassVar[dict]): The mapping between the string dtype
            and their torch sparse counterpart.
    """

    reader: ReaderConfig
    filtering: Dict[str, dict] = None
    models: Dict[str, dict]
    general: GeneralConfig = Field(default_factory=GeneralConfig)

    # Supported sparse precisions in numpy
    sparse_np_dtype: ClassVar[dict] = {
        "float32": np.float32,
        "float64": np.float64,
    }

    # Supported sparse precision in torch
    sparse_torch_dtype: ClassVar[dict] = {
        "float32": torch.float32,
        "float64": torch.float64,
    }

    @model_validator(mode="after")
    def config_validation(self) -> "WarpRecConfiguration":
        """This method checks if everything in the configuration file is missing or incorrect.

        Returns:
            WarpRecConfiguration: The validated configuration.

        Raises:
            FileNotFoundError: If the local file has not been found.
            ValueError: If any information between parts of the configuration file is inconsistent.
        """

        # Check if the local file exists
        if self.reader.reading_method == ReadingMethods.LOCAL:
            local_path: str = None
            if self.reader.local_path is not None:
                local_path = self.reader.local_path
            elif self.reader.split.local_path is not None:
                ext = self.reader.split.ext
                local_path = os.path.join(self.reader.split.local_path, "train" + ext)
            else:
                raise ValueError("Unsupported local source or missing local path.")

            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Training file not at {local_path}")

        # Load custom modules if specified
        load_custom_modules(self.general.custom_models)

        # Check if the filters have been set correctly
        if self.filtering is not None:
            labels = self.reader.labels.model_dump()
            for filter_name, filter_params in self.filtering.items():
                if filter_name.upper() not in filter_registry.list_registered():
                    raise ValueError(
                        f"Filter '{filter_name}' is not registered. These are the filters registered: {filter_registry.list_registered()}"
                    )
                try:
                    filter_registry.get(filter_name, **filter_params, **labels)
                except Exception as e:
                    raise ValueError(
                        f"Error initializing filter '{filter_name}' with these params {filter_params}: {e}"
                    )

        # Check if the precision is supported
        self.check_precision()

        return self

    def check_precision(self) -> None:
        """This method checks the precision passed through configuration.

        Raises:
            ValueError: If the precision is not supported or incorrect.
        """
        if self.general.precision not in self.sparse_np_dtype:
            raise ValueError(
                f"Custom dtype {self.general.precision} not supported as sparse data type."
            )

    def precision_numpy(self) -> np.dtype:
        """This method returns the precision that will be used for this experiment.

        Returns:
            np.dtype: The numpy precision requested.
        """
        return self.sparse_np_dtype[self.general.precision]

    def precision_torch(self) -> torch.dtype:
        """This method returns the precision that will be used for this experiment.

        Returns:
            torch.dtype: The torch precision requested.
        """
        return self.sparse_torch_dtype[self.general.precision]

    def get_filters(self) -> List[Filter]:
        """Returns the initialized filters based on the configuration.

        Returns:
            List[Filter]: A list of initialized filter instances.
        """
        if not self.filtering:
            return []
        labels = self.reader.labels.model_dump()
        return [
            filter_registry.get(filter_name, **filter_params, **labels)
            for filter_name, filter_params in self.filtering.items()
        ]


class TrainConfiguration(WarpRecConfiguration):
    """Definition of configuration, used to interact with the framework.

    This class defines the structure of the configuration file accepted by the framework.

    Attributes:
        writer (WriterConfig): Configuration of the writing process.
        splitter (SplittingConfig): Configuration of the splitting process.
        dashboard (DashboardConfig): Configuration of the dashboard process.
        evaluation (EvaluationConfig): Configuration of the evaluation process.
    """

    writer: WriterConfig
    splitter: SplittingConfig
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def train_validation(self) -> "TrainConfiguration":
        """This method checks if everything in the configuration file is missing or incorrect.

        Returns:
            TrainConfiguration: The validated configuration.

        Raises:
            ValueError: If any information between parts of the configuration file is inconsistent.
        """
        # Check if evaluation has been set up correctly
        if self.evaluation.full_evaluation_on_report:
            # Check if the validation metric is in the
            # metrics to evaluate
            val_metric, val_k = validation_metric(self.evaluation.validation_metric)

            if (
                val_metric not in self.evaluation.metrics
                or val_k not in self.evaluation.top_k
            ):
                raise ValueError(
                    "Full evaluation on report has been requested but the "
                    "validation metric it's not contained in the evaluation metrics. "
                    f"Validation metric: {self.evaluation.validation_metric} ."
                    f"Metrics to evaluate: {self.evaluation.metrics}. "
                    f"TopK to evaluate: {self.evaluation.top_k}."
                )

        # Parse and validate models
        self.check_precision()
        self.models = self.parse_models()

        return self

    def parse_models(self) -> dict:
        """This method parses the models and creates the correct data structures.

        Returns:
            dict: The dictionary containing all the models and their parameters.

        Raises:
            ValueError: If a model requires side information and they have not been provided.
        """
        parsed_models = {}

        for model_name, model_data in self.models.items():
            model_class: RecomModel
            if model_name.upper() not in model_registry.list_registered():
                logger.negative(
                    f"The model {model_name} is not registered in the model registry. "
                    "The model will not be loaded and will not be available for training. "
                    "Check the configuration file."
                )
                continue
            elif (
                model_name.upper() in model_registry.list_registered()
                and model_name.upper() not in params_registry.list_registered()
            ):
                logger.attention(
                    f"The model {model_name} is registered in the model registry, but not in parameter registry. "
                    "The model will be loaded but the hyperparameter will not be validated."
                )
                model_class = RecomModel(**model_data)
                parsed_models[model_name] = model_class.model_dump()
                continue
            else:
                model_class = params_registry.get(model_name, **model_data)

                if model_class.need_side_information and self.reader.side is None:
                    raise ValueError(
                        f"The model {model_name} requires side information to be provided, "
                        "but none have been provided. Check the configuration file."
                    )

                # Check if there is at least one valid combination
                model_class.validate_all_combinations()

                # Check if the model requires timestamp
                if model_class.need_timestamp:
                    logger.attention(
                        f"The model {model_name} requires timestamps to work properly, "
                        "be sure that your dataset contains them."
                    )

                # Extract model train parameters, removing the meta infos
                model_data = {
                    k: (
                        [v]
                        if not isinstance(v, list) and v is not None and k != "meta"
                        else v
                    )
                    for k, v in model_data.items()
                }

                parsed_models[model_name] = model_class.model_dump()

        return parsed_models


class DesignConfiguration(WarpRecConfiguration):
    """Definition of design pipeline configuration, used to test custom models.

    Attributes:
        splitter (SplittingConfig): Configuration of the splitting process.
        evaluation (EvaluationConfig): Configuration of the evaluation process.
    """

    splitter: SplittingConfig = None
    evaluation: EvaluationConfig


def load_train_configuration(path: str) -> TrainConfiguration:
    """This method reads the train configuration file and returns
        a TrainConfiguration object.

    Args:
        path (str): The path to the configuration file.

    Returns:
        TrainConfiguration: The configuration object created from the configuration file.
    """
    logger.msg(f"Reading train configuration file in: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    logger.msg("Reading process completed correctly.")
    return TrainConfiguration(**data)


def load_design_configuration(path: str) -> DesignConfiguration:
    """This method reads the train configuration file and returns
        a DesignConfiguration object.

    Args:
        path (str): The path to the configuration file.

    Returns:
        DesignConfiguration: The configuration object created from the configuration file.
    """
    logger.msg(f"Reading design configuration file in: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    logger.msg("Reading process completed correctly.")
    return DesignConfiguration(**data)


def load_callback(
    callback_config: WarpRecCallbackConfig, *args: Any, **kwargs: Any
) -> WarpRecCallback:
    """Dynamically loads and initializes a custom WarpRecCallback class
    based on the provided configuration.

    This function assumes that `callback_config` has already been validated
    via Pydantic, ensuring that the module path and class name are correct,
    and that the class exists and is a subclass of `WarpRecCallback`.

    Args:
        callback_config (WarpRecCallbackConfig): The Pydantic configuration object
            for the custom callback.
        *args (Any): Additional positional arguments to pass to the callback's constructor.
        **kwargs (Any): Additional keyword arguments to pass to the callback's constructor.

    Returns:
        WarpRecCallback: An instance of the custom callback, or None if no
            custom callback is specified in the configuration.

    Raises:
        RuntimeError: If an unexpected error occurs during loading or initialization,
            given that prior validation should have prevented most errors.
    """
    if (
        callback_config is None
        or callback_config.callback_path is None
        or callback_config.callback_name is None
    ):
        return WarpRecCallback()  # Empty callback used for consistency

    module_path = Path(callback_config.callback_path)
    class_name = callback_config.callback_name

    # Save the original sys.path to restore it afterwards
    original_sys_path = sys.path[:]

    try:
        # Add the module's directory to sys.path to allow for internal imports
        module_dir = module_path.parent
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None:
            raise RuntimeError(
                f"Could not load spec for module: {module_path}. "
                f"This should not happen after validation."
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the class from the module
        callback_class = getattr(module, class_name)

        # Initialize and return the callback instance
        return callback_class(*args, **kwargs)

    except Exception as e:
        # Catch any residual errors, though validation should prevent most
        raise RuntimeError(
            f"Unexpected error during initialization of callback '{class_name}' "
            f"from '{module_path}': {e}"
        ) from e
    finally:
        # Restore sys.path to avoid side-effects
        sys.path = original_sys_path
