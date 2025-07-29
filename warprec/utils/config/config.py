import os
import sys
import importlib
from pathlib import Path
from typing import Tuple, ClassVar, Dict, Any

import yaml
import numpy as np
import torch
from pydantic import BaseModel, field_validator, model_validator, Field
from warprec.utils.helpers import load_custom_modules
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
from warprec.utils.enums import RatingType, SplittingStrategies, ReadingMethods
from warprec.utils.registry import model_registry, params_registry
from warprec.utils.logger import logger


class Configuration(BaseModel):
    """Definition of configuration, used to interact with the framework.

    This class defines the structure of the configuration file accepted by the framework.

    Attributes:
        reader (ReaderConfig): Configuration of the reading process.
        writer (WriterConfig): Configuration of the writing process.
        splitter (SplittingConfig): Configuration of the splitting process.
        dashboard (DashboardConfig): Configuration of the dashboard process.
        models (Dict[str, dict]): The dictionary containing model information
            in the format {model_name: dict{param_1: value, param_2: value, ...}, ...}
        evaluation (EvaluationConfig): Configuration of the evaluation process.
        general (GeneralConfig): General configuration of the experiment
        sparse_np_dtype (ClassVar[dict]): The mapping between the string dtype
            and their numpy sparse counterpart.
        sparse_torch_dtype (ClassVar[dict]): The mapping between the string dtype
            and their torch sparse counterpart.
        need_session_based_information (ClassVar[bool]): Wether or not the experiments
            will be conducted on session data.
    """

    reader: ReaderConfig
    writer: WriterConfig
    splitter: SplittingConfig = None
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    models: Dict[str, dict]
    evaluation: EvaluationConfig
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

    # Track if session-based information is needed
    need_session_based_information: ClassVar[bool] = False

    @field_validator("splitter", mode="before")
    @classmethod
    def check_splitter(cls, v: SplittingConfig) -> SplittingConfig:
        """Validate splitter."""
        if v is None:
            return SplittingConfig()
        return v

    @field_validator("general", mode="before")
    @classmethod
    def check_general(cls, v: GeneralConfig) -> GeneralConfig:
        """Validate general configuration."""
        if v is None:
            return GeneralConfig()
        return v

    @model_validator(mode="after")
    def config_validation(self) -> "Configuration":
        """This method checks if everything in the configuration file is missing or incorrect.

        When the configuration passes this check, everything should be good to go.

        Returns:
            Configuration: The validated configuration.

        Raises:
            FileNotFoundError: If the local file has not been found.
            ValueError: If any information between parts of the configuration file is inconsistent.
        """

        # Check if the local file exists and is correctly
        # formatted
        if self.reader.reading_method == ReadingMethods.LOCAL:
            _local_path: str = None
            _sep: str = None
            _has_header: bool = None
            if self.reader.local_path is not None:
                _local_path = self.reader.local_path
                _sep = self.reader.sep
                _has_header = self.reader.header
            elif self.reader.split.local_path is not None:
                _ext = self.reader.split.ext
                _local_path = os.path.join(self.reader.split.local_path, "train" + _ext)
                _sep = self.reader.split.sep
                _has_header = self.reader.split.header
            else:
                raise ValueError("Unsupported local source or missing local path.")

            if not os.path.exists(_local_path):
                raise FileNotFoundError(f"Training file not at {_local_path}")

            # Read the header of file to later check
            with open(_local_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            _header = first_line.strip().split(_sep)

            # If the source file should have header, we check
            # if the column names are present.
            if _has_header:
                # Define column names to read after, if the input file
                # contains more columns this is more efficient.
                _column_names = [
                    self.reader.labels.user_id_label,
                    self.reader.labels.item_id_label,
                ]
                if self.reader.rating_type == RatingType.EXPLICIT:
                    # In case the RatingType is explicit, we add the
                    # score label and read scores from the source.
                    _column_names.append(self.reader.labels.rating_label)
                if self.splitter is not None and self.splitter.strategy in [
                    SplittingStrategies.TEMPORAL_HOLDOUT,
                    SplittingStrategies.TEMPORAL_LEAVE_K_OUT,
                ]:
                    # In case the SplittingStrategy is temporal, we add the
                    # timestamp label and read timestamps from the source.
                    _column_names.append(self.reader.labels.timestamp_label)

                # Check if column name defined in config are present in the header of the local file
                if not set(_column_names).issubset(set(_header)):
                    error_msg = (
                        "Column labels required do not match with the "
                        "column names found in the local file. "
                        f"Expected: {', '.join(_column_names)}. "
                        f"Found: {', '.join(_header)}. "
                    )

                    raise ValueError(error_msg)

                # Updating reader config with the column names
                ReaderConfig.column_names = _column_names
            else:
                # If the header is not present, we check if the number of columns
                # in the file matches the number of columns expected.
                _expected_columns = 2  # user_id and item_id
                _column_names = [
                    self.reader.labels.user_id_label,
                    self.reader.labels.item_id_label,
                ]
                if self.reader.rating_type == RatingType.EXPLICIT:
                    _expected_columns += 1
                    _column_names.append(self.reader.labels.rating_label)
                if self.splitter.strategy in [
                    SplittingStrategies.TEMPORAL_HOLDOUT,
                    SplittingStrategies.TEMPORAL_LEAVE_K_OUT,
                ]:
                    _expected_columns += 1
                    _column_names.append(self.reader.labels.timestamp_label)
                if len(_header) != _expected_columns:
                    raise ValueError(
                        "The number of columns in the local file does not match "
                        "the number of columns expected. Check the configuration."
                    )

                # Updating reader config with the column names
                ReaderConfig.column_names = _column_names

        # Check if experiment has been set up correctly
        if not self.writer.setup_experiment:
            for model_name, model_data in self.models.items():
                if model_data["meta"]["save_model"]:
                    raise ValueError(
                        f"You are trying to save the model state for {model_name} model but "
                        "experiment must be setup first. Set setup_experiment to True."
                    )
            if self.writer.save_split:
                raise ValueError(
                    "You are trying to save the splits but experiment must be "
                    "setup first. Set setup_experiment to True."
                )
            if self.evaluation.save_evaluation:
                raise ValueError(
                    "You are trying to save the evaluation but experiment must be "
                    "setup first. Set setup_experiment to True."
                )

        load_custom_modules(self.general.custom_models)

        # Final checks and parsing
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
                logger.negative(
                    f"The model {model_name} is registered in the model registry, but not in parameter registry. "
                    "The model will not be loaded and will not be available for training. "
                    "Check the configuration file."
                )
                continue
            else:
                model_class: RecomModel = params_registry.get(model_name, **model_data)

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

                # If at least one model is a sequential model, then
                # we set the flag for session-based information
                if model_class.need_timestamp:
                    Configuration.need_session_based_information = True

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

    def validation_metric(self, val_metric: str) -> Tuple[str, int]:
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


def load_yaml(path: str) -> Configuration:
    """This method reads the configuration file and returns a Configuration object.

    Args:
        path (str): The path to the configuration file.

    Returns:
        Configuration: The configuration object created from the configuration file.
    """
    logger.msg(f"Reading configuration file in: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    logger.msg("Reading process completed correctly.")
    return Configuration(**data)


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
