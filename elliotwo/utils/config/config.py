import os
from typing import Tuple, ClassVar, Dict

import yaml
import numpy as np
import torch
from pydantic import BaseModel, field_validator, model_validator
from elliotwo.utils.config import (
    GeneralConfig,
    ReaderConfig,
    WriterConfig,
    SplittingConfig,
    RecomModel,
    EvaluationConfig,
)
from elliotwo.utils.enums import RatingType, SplittingStrategies, ReadingMethods
from elliotwo.utils.registry import params_registry
from elliotwo.utils.logger import logger


class Configuration(BaseModel):
    """Definition of configuration, used to interact with the framework.

    This class defines the structure of the configuration file accepted by the framework.

    Attributes:
        reader (ReaderConfig): Configuration of the reading process.
        writer (WriterConfig): Configuration of the writing process.
        splitter (SplittingConfig): Configuration of the splitting process.
        models (Dict[str, dict]): The dictionary containing model information
            in the format {model_name: dict{param_1: value, param_2: value, ...}, ...}
        evaluation (EvaluationConfig): Configuration of the evaluation process.
        general (GeneralConfig): General configuration of the experiment
        sparse_np_dtype (ClassVar[dict]): The mapping between the string dtype
            and their numpy sparse counterpart.
        sparse_torch_dtype (ClassVar[dict]): The mapping between the string dtype
            and their torch sparse counterpart.
    """

    reader: ReaderConfig
    writer: WriterConfig
    splitter: SplittingConfig = None
    models: Dict[str, dict]
    evaluation: EvaluationConfig
    general: GeneralConfig = None

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

        # Check if file exists
        if (
            self.reader.reading_method == ReadingMethods.LOCAL
            and self.reader.local_path
        ):
            _local_path = self.reader.local_path
            if not os.path.exists(_local_path):
                raise FileNotFoundError(
                    f"Configuration file not found at {_local_path}."
                )
            _sep = self.reader.sep
            # Read the header of file to later check
            with open(_local_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
            _header = first_line.strip().split(_sep)

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
            if self.splitter.strategy in [
                SplittingStrategies.TEMPORAL_HOLDOUT,
                SplittingStrategies.TEMPORAL_LEAVE_K_OUT,
            ]:
                # In case the SplittingStrategy is temporal, we add the
                # timestamp label and read timestamps from the source.
                _column_names.append(self.reader.labels.timestamp_label)

            # Check if column name defined in config are present in the header of the local file
            if not set(_column_names).issubset(set(_header)):
                raise ValueError(
                    "Column labels required do not match with the "
                    "column names found in the local file."
                )

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
            model_class: RecomModel = params_registry.get(model_name, **model_data)

            if model_class.need_side_information and self.reader.side is None:
                raise ValueError(
                    f"The model {model_name} requires side information to be provided, "
                    "but none have been provided. Check the configuration file."
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


def check_separator(sep: str) -> str:
    """This method checks the separator, if it's not in a correct format
        then it is set to the default separator.

    Args:
        sep (str): The separator to check.

    Returns:
        str: The validated separator.
    """
    try:
        sep = sep.encode().decode("unicode_escape")
    except UnicodeDecodeError:
        logger.negative(
            f"The string {sep} is not a valid separator. Using default separator {'\t'}."
        )
        sep = "\t"
    return sep
