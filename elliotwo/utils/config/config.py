import os
from typing import Tuple, List, ClassVar, Dict
from copy import deepcopy

import yaml
import numpy as np
import torch
from pydantic import BaseModel, field_validator, model_validator
from elliotwo.utils.config import (
    GeneralConfig,
    DataConfig,
    SplittingConfig,
    RecomModel,
    EvaluationConfig,
)
from elliotwo.utils.enums import RatingType, SplittingStrategies
from elliotwo.utils.registry import params_registry, search_space_registry
from elliotwo.utils.logger import logger


class Configuration(BaseModel):
    """Definition of configuration, used to interact with the framework.

    This class defines the structure of the configuration file accepted by the framework.

    Attributes:
        data (DataConfig): Configuration of the data loading process.
        splitter (SplittingConfig): Configuration of the splitting process.
        models (Dict[str, dict]): The dictionary containing model information
            in the format {model_name: dict{param_1: value, param_2: value, ...}, ...}
        evaluation (EvaluationConfig): Configuration of the evaluation process.
        general (GeneralConfig): General configuration of the experiment
        column_map_dtype (ClassVar[dict]): The mapping between the string dtype
            and their numpy counterpart.
        sparse_np_dtype (ClassVar[dict]): The mapping between the string dtype
            and their numpy sparse counterpart.
        sparse_torch_dtype (ClassVar[dict]): The mapping between the string dtype
            and their torch sparse counterpart.
    """

    data: DataConfig
    splitter: SplittingConfig = None
    models: Dict[str, dict]
    evaluation: EvaluationConfig
    general: GeneralConfig = None

    # Supported dtype
    column_map_dtype: ClassVar[dict] = {
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "float32": np.float32,
        "float64": np.float64,
        "string": np.str_,
    }

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
        _local_path = self.data.local_path

        # Check if file exists
        if _local_path:
            if not os.path.exists(_local_path):
                raise FileNotFoundError(
                    f"Configuration file not found at {_local_path}."
                )
            _sep = self.data.sep
            # Read the header of file to later check
            with open(_local_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            _header = first_line.strip().split(_sep)

            # Define column names to read after, if the input file
            # contains more columns this is more efficient.
            _column_names = [
                self.data.labels.user_id_label,
                self.data.labels.item_id_label,
            ]
            if self.data.rating_type == RatingType.EXPLICIT:
                # In case the RatingType is explicit, we add the
                # score label and read scores from the source.
                _column_names.append(self.data.labels.rating_label)
            if self.splitter.strategy == SplittingStrategies.TEMPORAL:
                # In case the SplittingStrategy is temporal, we add the
                # timestamp label and read timestamps from the source.
                _column_names.append(self.data.labels.timestamp_label)

            # Check if column name defined in config are present in the header of the local file
            if not set(_column_names).issubset(set(_header)):
                raise ValueError(
                    "Column labels required do not match with the "
                    "column names found in the local file."
                )

        # Check if experiment has been set up correctly
        if not self.general.setup_experiment:
            for model_name, model_data in self.models.items():
                if model_data["meta"]["save_model"]:
                    raise ValueError(
                        f"You are trying to save the model state for {model_name} model but "
                        "experiment must be setup first. Set setup_experiment to True."
                    )
            if self.splitter.save_split:
                raise ValueError(
                    "You are trying to save the splits but experiment must be "
                    "setup first. Set setup_experiment to True."
                )
            if self.evaluation.save_evaluation:
                raise ValueError(
                    "You are trying to save the evaluation but experiment must be "
                    "setup first. Set setup_experiment to True."
                )

        # Final checks and parsing
        self.check_column_dtype()
        self.check_precision()
        self.models = self.parse_models()

        return self

    def parse_models(self) -> dict:
        """This method parses the models and creates the correct data structures.

        Returns:
            dict: The dictionary containing all the models and their parameters.
        """
        parsed_models = {}

        for model_name, model_data in self.models.items():
            model_class: RecomModel = params_registry.get(model_name, **model_data)

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

    def check_column_dtype(self) -> None:
        """This method validates the custom dtype passed with the configuration file.

        Raises:
            ValueError: If the dtype are not supported or incorrect.
        """
        for dtype_str in self.data.dtype.model_dump().values():
            if dtype_str not in self.column_map_dtype:
                raise ValueError(
                    f"Custom dtype {dtype_str} not supported as a column data type."
                )

    def column_dtype(self) -> List[np.dtype]:
        """This method will parse the dtype from the string forma to their numpy counterpart.

        Returns:
            List[np.dtype]: A list containing the dtype to use for data loading.
        """
        return [
            self.column_map_dtype[dtype_str]
            for dtype_str in self.data.dtype.model_dump().values()
        ]

    def column_names(self) -> List[str]:
        """This method returns the names of the column passed through configuration.

        Returns:
            List[str]: The list of column names.
        """
        return list(self.data.labels.model_dump().values())

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


def parse_params(params: dict) -> dict:
    """This method parses the parameters of a model.

    From simple lists it creates the correct data format for
    Ray Tune hyperparameter optimization.

    Args:
        params (dict): The parameters of the model.

    Returns:
        dict: The parameters in the Ray Tune format.
    """
    tune_params = {}
    params_copy = deepcopy(params)
    params_copy.pop("meta")
    params_copy.pop("optimization")
    for k, v in params_copy.items():
        tune_params[k] = search_space_registry.get(v[0])(*v[1:])

    return tune_params


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
