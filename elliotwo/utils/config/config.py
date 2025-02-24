import os
from typing import Tuple, Type, List, Optional, ClassVar, Dict
from abc import abstractmethod

import yaml
import numpy as np
import torch
from ray import tune
from pydantic import BaseModel, Field, field_validator, model_validator
from elliotwo.utils.enums import RatingType, SplittingStrategies
from elliotwo.utils.logger import logger


class Labels(BaseModel):
    """Definition of the label sub-configuration.

    This class reads and optionally overrides the default labels of important data.

    Attributes:
        user_id_label (Optional[str]): Name of the user ID label. Defaults to 'user_id'.
        item_id_label (Optional[str]): Name of the item ID label. Defaults to 'item_id'.
        rating_label (Optional[str]): Name of the rating label. Defaults to 'rating'.
        timestamp_label (Optional[str]): Name of the timestamp label. Defaults to 'timestamp'.
    """

    user_id_label: Optional[str] = "user_id"
    item_id_label: Optional[str] = "item_id"
    rating_label: Optional[str] = "rating"
    timestamp_label: Optional[str] = "timestamp"


class CustomDtypes(BaseModel):
    """Definition of the custom dtypes sub-configuration.

    This class reads and optionally ovverrides default labels of important data.

    Attributes:
        user_id_type (Optional[str]): The dtypes to format the user_id column.
        item_id_type (Optional[str]): The dtypes to format the item_id column.
        rating_type (Optional[str]): The dtypes to format the rating column.
        timestamp_type (Optional[str]): The dtypes to format the timestamp column.
    """

    user_id_type: Optional[str] = "int32"
    item_id_type: Optional[str] = "int32"
    rating_type: Optional[str] = "float32"
    timestamp_type: Optional[str] = "int32"


class DataConfig(BaseModel):
    """Definition of the data configuration part of the configuration file.

    Attributes:
        dataset_name (str): Name of the dataset.
        loading_strategy (str): The strategy to use to load the data. Can be 'dataset' or 'split'.
        data_type (str): The type of data to be loaded. Can be 'transaction'.
        local_path (Optional[str]): Path to the file containing the transaction data.
        split_dir (Optional[str]): The directory where the splits are saved.
        experiment_path (Optional[str]): The local experiment path.
        sep (Optional[str]): Custom separator for the file containing the transaction data.
        rating_type (RatingType): The type of rating to be used. If 'implicit' is chosen, \
            the the reader will not look for a score.
        batch_size (Optional[int]): The batch size to be used during the reading process. \
            If None is chosen, the data will be read in one pass.
        labels (Labels): The labels sub-configuration. Defaults to Labels default values.
        dtypes (CustomDtypes): The list of column dtypes.
    """

    dataset_name: str
    loading_strategy: str
    data_type: str
    local_path: Optional[str] = None
    split_dir: Optional[str] = None
    experiment_path: Optional[str] = "./experiments/"
    sep: Optional[str] = ","
    rating_type: RatingType
    batch_size: Optional[int] = None
    labels: Labels = Field(default_factory=Labels)
    dtypes: CustomDtypes = Field(default_factory=CustomDtypes)

    @field_validator("loading_strategy")
    @classmethod
    def check_loading_strategy(cls, v: str):
        """Validates the loading strategy.

        Raise:
            ValueError: If the loading strategy is incorrect or not supported.
        """
        supported_strategies = ["dataset", "split"]
        if v not in supported_strategies:
            raise ValueError(
                f"Loading strategy {v} not supported. Supported strategies: {supported_strategies}."
            )
        return v

    @field_validator("data_type")
    @classmethod
    def check_data_type(cls, v: str):
        """Validates the data type.

        Raise:
            ValueError: If the data type is incorrect or not supported.
        """
        supported_data_types = ["transaction"]
        if v not in supported_data_types:
            raise ValueError(
                f"Data type {v} not supported. Supported data types: {supported_data_types}."
            )
        return v

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator.

        Raise:
            UnicodeDecodeError: If the separator is not correct.
        """
        try:
            v = v.encode().decode("unicode_escape")
        except UnicodeDecodeError:
            logger.negative(
                f"The string {v} is not a valid separator. Using default separator {','}."
            )
            v = ","
        return v

    @model_validator(mode="after")
    def check_data(self):
        """This method checks if the required informations have been passed to the configuration.

        Raise:
            ValueError: If an important field has not been filled with the correct information.
        """
        # ValueError checks
        if self.loading_strategy == "split" and not self.split_dir:
            raise ValueError(
                "You have chosen split loading strategy but the split_dir \
                    field has not been filled."
            )
        if self.loading_strategy == "dataset" and not self.local_path:
            raise ValueError(
                "You have chosen dataset loading strategy but the local_path \
                    field has not been filled."
            )

        # Attention checks
        if self.loading_strategy == "split" and self.local_path:
            logger.attention(
                "You have chosen split loading strategy but the local_path field \
                    has been filled. Check your configuration file for possible errors."
            )
        if self.loading_strategy == "dataset" and self.split_dir:
            logger.attention(
                "You have chosen dataset loading strategy but the split_dir field \
                    has been filled. Check your configuration file for possible errors."
            )
        return self


class SplittingConfig(BaseModel):
    """Definition of the splitting configuration part of the configuration file.

    This class reads all the side information about the experiment from the configuration file.

    Attributes:
        strategy (Optional[SplittingStrategies]): The splitting strategy to be used to split data.
        validation (Optional[bool]): Whether or not to create a validation \
            split during the splitting process.
        ratio (Optional[List[float]]): The ratios of the splitting to be created.
        k (Optional[int]): The number of examples to remove during leave-k-out strategy.
        save_split (Optional[bool]): Whether or not to save the splits created for later use.
    """

    strategy: Optional[SplittingStrategies] = SplittingStrategies.NONE
    validation: Optional[bool] = False
    ratio: Optional[List[float]] = None
    k: Optional[int] = None
    save_split: Optional[bool] = False

    @field_validator("ratio")
    @classmethod
    def check_sum_to_one(cls, v):
        """Validates the ratios.

        Raise:
            ValueError: If the list length is different from 2 or 3.
            ValueError: The sum of the values isn't 1.
        """
        if v is not None:
            if len(v) not in [2, 3]:
                raise ValueError(
                    "List must be of length 2 in case of train/test split or 3 \
                        in case of train/val/test split."
                )
            if not abs(sum(v) - 1.0) < 1e-6:  # Slight tolerance for the sum
                raise ValueError(f"The sum of ratios must be 1. Received sum: {sum(v)}")
        return v

    @model_validator(mode="after")
    def check_dependecies(self):
        """This method checks if the required informations have been passed to the configuration.

        Raise:
            ValueError: If an important field has not been filled with the correct information.
            Warning: If a field that will not be used during the experiment has been filled.
        """
        # ValueError checks
        if self.strategy in [SplittingStrategies.RANDOM, SplittingStrategies.TEMPORAL]:
            if self.ratio is None:
                raise ValueError(
                    f"You have chosen {self.strategy.value} splitting but \
                        the ratio field has not been filled."
                )
            _expected_ratio = 3 if self.validation else 2
            if len(self.ratio) != _expected_ratio:
                raise ValueError(
                    "The ratio and the number of split expectd \
                        do not match. Check if validation set parameter \
                            has been set or if ratio values are correct."
                )

        # Attention checks
        if (
            self.strategy in [SplittingStrategies.RANDOM, SplittingStrategies.TEMPORAL]
            and self.k
        ):
            logger.attention(
                f"You have filled the k field but the splitting strategy \
                    has been set to {self.strategy.value}. Check your \
                        configuration file for possible errors."
            )
        if self.strategy == SplittingStrategies.LEAVE_ONE_OUT and self.ratio:
            logger.attention(
                "You have filled the ratio field but splitting strategy \
                    has been set to leave-one-out. Check your \
                        configuration file for possible errors."
            )

        return self


class Meta(BaseModel):
    """Definition of the Meta-information sub-configuration of a RecommenderModel.

    Attributes:
        save_model (Optional[bool]): Whether save or not the model state after training.
        load_from (Optional[str]): The path where a previos model state has been saved.
    """

    save_model: Optional[bool] = False
    load_from: Optional[str] = None


class RecomModel(BaseModel):
    """Definition of a RecommendationModel configuration. All models must extend this class.

    Attributes:
        meta (Meta): The meta-information about the model. Defaults to Meta default values.
    """

    meta: Meta = Field(default_factory=Meta)

    @abstractmethod
    def get_params(self, param_dict: dict) -> dict:
        """This method transforms the parameters passed to the model
        in their correct format, to be ingested by Ray Tune.

        Every model should implement their own way of parsing the parameters in the correct format.

        Args:
            param_dict (dict): The dictionary containing the parameters to parse.

        Returns:
            dict: The dictionary with parsed parameters for Ray Tune.
        """


class EASE(RecomModel):
    """Definition of the model EASE.

    Attributes:
        l2 (Optional[List[float]]): List of values that l2 regularization can take.
        implementation (Optional[List[str]]): List of different implementation to test out.
    """

    l2: Optional[List[float]] = [1.0, 2.0]
    implementation: Optional[List[str]] = ["classic"]

    @field_validator("implementation")
    @classmethod
    def check_implementation(cls, v):
        """Validates implementation.

        Raise:
            ValueError: If the implementation is not supported.
        """
        if not isinstance(v, list):
            v = [v]
        for imp in v:
            if imp not in ["classic", "elliot"]:
                raise ValueError(f"Implementation {imp} not supported by model EASE.")
        return v

    @field_validator("l2")
    @classmethod
    def check_l2(cls, v):
        """Validates the l2 regularization.

        Raise:
            ValueErrro: If the l2 is not a range compatible with hyperopt formulation.
        """
        if not isinstance(v, list):
            raise ValueError(
                "L2 value must be a list that represents the min and max value to explore."
            )
        if len(v) != 2:
            raise ValueError(
                "L2 value must be a list that represents the min and max value to explore."
            )
        return v

    def get_params(self, param_dict: dict) -> dict:
        return {
            "l2": tune.uniform(param_dict["l2"][0], param_dict["l2"][1]),
            "implementation": tune.choice(param_dict["implementation"]),
        }


class Slim(RecomModel):
    """Definition of the model Slim.

    Attributes:
        l1 (Optional[List[float]]): List of values that l1 regularization can take.
        alpha (Optional[List[float]]): List of values that alpha can take.
    """

    l1: Optional[List[float]]
    alpha: Optional[List[float]]

    def get_params(self, param_dict: dict) -> dict:
        pass


MODEL_REGISTRY: Dict[str, Type[RecomModel]] = {"EASE": EASE, "Slim": Slim}


class EvaluationConfig(BaseModel):
    """Definition of Evaluation configuration.

    Attributes:
        top_k (List[int]): List of cutoffs to evaluate.
        metrics (List[str]): List of metrics to compute during evaluation.
        save_evaluation (Optional[bool]): Wether or not to save the evaluation.
    """

    top_k: List[int]
    metrics: List[str]
    save_evaluation: Optional[bool] = True

    @field_validator("metrics")
    @classmethod
    def metrics_validator(cls, v):
        """Validate metrics.

        Raise:
            ValueError: If the metric is not present in the METRICS_REGISTRY.
        """
        for metric in v:
            if metric not in METRICS_REGISTRY:
                raise ValueError(f"Metric {metric} not supported.")
        return v


METRICS_REGISTRY = ["NDCG", "Precision", "Recall", "HitRate"]


class GeneralRecommendation(BaseModel):
    """Definition of recommendation informations.

    Attributes:
        save_recs (Optional[bool]): Flag for recommendation saving. Defaults to False.
        sep (Optional[str]): Custom separator to use during recomendation saving. Defaults to ','.
        ext (Optional[str]): Custom extension. Defaults to '.csv'.
    """

    save_recs: Optional[bool] = False
    sep: Optional[str] = ","
    ext: Optional[str] = ".csv"

    @field_validator("sep")
    @classmethod
    def check_sep(cls, v: str):
        """Validates the separator.

        Raise:
            UnicodeDecodeError: If the separator is not correct.
        """
        try:
            v = v.encode().decode("unicode_escape")
        except UnicodeDecodeError:
            logger.negative(
                f'The string {v} is not a valid separator. Using default separator ",".'
            )
            v = ","
        return v


class GeneralConfig(BaseModel):
    """Definition of the general configuration part of the configuration file.

    This class reads all the side information about the experiment from the configuration file.

    Attributes:
        seed (Optional[int]): The seed that will be used during the experiment for reproducibility.
        float_digits (Optional[int]): The number of floating point digits to show on console.
        device (Optional[str]): The device that will be used for most operations.
        validation_metric (Optional[str]): The validation metric to use,
            in the format of metric_name@top_k.
        precision (Optional[str]): The precision to use during computation.
        max_evals (Optional[int]): The maximum number of evaluations to compute with hyperopt.
        recommendation (Optional[GeneralRecommendation]): The general informations
            about the recommendation.
        setup_experiment (Optional[bool]): Wether or not to setup the experiment ambient.
    """

    seed: Optional[int] = 42
    float_digits: Optional[int] = 16
    device: Optional[str] = "cpu"
    validation_metric: Optional[str] = "nDCG@5"
    precision: Optional[str] = "float32"
    max_evals: Optional[int] = 10
    recommendation: Optional[GeneralRecommendation] = Field(
        default_factory=GeneralRecommendation
    )
    setup_experiment: Optional[bool] = True

    @field_validator("device")
    @classmethod
    def check_device(cls, v):
        """Validate device.

        Raise:
            ValueError: If the device is not in the correct format.
        """
        if v in ("cuda", "cpu"):
            return v
        if v.startswith("cuda:"):
            parts = v.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                return v
        raise ValueError(f'Device {v} is not supported. Use "cpu" or "cuda[:index]".')

    @field_validator("validation_metric")
    @classmethod
    def check_validation_metric(cls, v):
        """Validate validation metric.

        Raise:
            ValueError: If the validation metric is not in the correct format.
        """
        if "@" not in v:
            raise ValueError(
                f"Validation metric {v} not valid. Validation metric \
                    should be defined as: metric_name@top_k."
            )
        if v.count("@") > 1:
            raise ValueError(
                "Validation metric contains more than one @, check your configuration file."
            )
        metric, top_k = v.split("@")
        if metric not in METRICS_REGISTRY:
            raise ValueError(f"The metric {metric} if not present in METRIC_REGISTRY.")
        if not top_k.isnumeric():
            raise ValueError(
                "Validation metric should be provided with a top_k number."
            )
        return v

    @model_validator(mode="after")
    def model_validation(self):
        """This method validates the General configuration.

        Raise:
            ValueError: If some values are inconsistent in the configuration file.
        """
        if self.recommendation.save_recs and not self.setup_experiment:  # pylint: disable=no-member
            raise ValueError(
                "You are trying to save the recommendations without "
                "setting up the directoy. Set setup_experiment to True."
            )
        return self


class Configuration(BaseModel):
    """Definition of configuration, used to interact with the framework.

    This class defines the structure of the configuration file accepted by the framework.

    Attributes:
        data (DataConfig): Configuration of the dataloading process.
        splitter (SplittingConfig): Configuration of the splitting process.
        models (Dict[str, dict]): The dictionary containing model informations \
            in the format {model_name: dict{param_1: value, param_2: value, ...}, ...}
        evaluation (EvaluationConfig): Configuration of the evaluation process.
        general (GeneralConfig): General configuration of the experiment
        column_map_dtype (ClassVar[dict]): The mapping between the string dtypes \
            and their numpy counterpart.
        sparse_np_dtype (ClassVar[dict]): The mapping between the string dtypes \
            and their numpy sparse counterpart.
        sparse_torch_dtype (ClassVar[dict]): The mapping between the string dtypes \
            and their torch sparse counterpart.
    """

    data: DataConfig
    splitter: SplittingConfig = None
    models: Dict[str, dict]
    evaluation: EvaluationConfig
    general: GeneralConfig = None

    # Supported dtypes
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
        """Validate splitter.

        Args:
            v (SplittingConfig): The SplittingConfig object to validate.

        Returns:
            SplittingConfig: The SplittingConfig object if the value is not None.
        """
        if v is None:
            return SplittingConfig()
        return v

    @field_validator("general", mode="before")
    @classmethod
    def check_general(cls, v: GeneralConfig) -> GeneralConfig:
        """Validate splitter.

        Args:
            v (GeneralConfig): The GeneralConfig object to validate.

        Returns:
            GeneralConfig: The GeneralConfig object if the value is not None.
        """
        if v is None:
            return GeneralConfig()
        return v

    @model_validator(mode="after")
    def config_validation(self):
        """This method checks if everything in the configuration file is missing or incorrect.

        When the configuration passes this check, everything should be good to go.

        Raise:
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
                # In case the RatyingType is explicit, we add the
                # score label and read scores from the source.
                _column_names.append(self.data.labels.rating_label)
            if self.splitter.strategy == SplittingStrategies.TEMPORAL:
                # In case the SplittingStrategy is temporale, we add the
                # timestamp label and read timestamps from the source.
                _column_names.append(self.data.labels.timestamp_label)

            # Check if column name defined in config are present in the header of the local file
            if not set(_column_names).issubset(set(_header)):
                raise ValueError(
                    "Column labels required do not match with the \
                        column names found in the local file."
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
            if self.evaluation.save_valuation:
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
            dict: The dictionary containig all the models and their parameters.

        Raises:
            ValueError: If the model is not present in the MODEL_REGISTRY.
        """
        parsed_models = {}

        for model_name, model_data in self.models.items():
            model_class = MODEL_REGISTRY.get(model_name)
            if not model_class:
                raise ValueError(f"Model {model_name} not found in MODEL_REGISTRY.")

            model_data = {
                k: (
                    [v]
                    if not isinstance(v, list) and v is not None and k != "meta"
                    else v
                )
                for k, v in model_data.items()
            }

            parsed_models[model_name] = model_class(**model_data).model_dump()
        return parsed_models

    def check_column_dtype(self) -> None:
        """This method validates the custom dtypes passed with the configuration file.

        Raise:
            ValueError: If the dtypes are not supported or incorrect.
        """
        for dtype_str in self.data.dtypes.model_dump().values():
            if dtype_str not in self.column_map_dtype:
                raise ValueError(
                    f"Custom dtype {dtype_str} not supported as a column data type."
                )

    def column_dtype(self) -> List[np.dtype]:
        """This method will parse the dtypes from the string forma to their numpy counterpart.

        Returns:
            List[np.dtype]: A list containing the dtypes to use for dataloading.
        """
        return [
            self.column_map_dtype[dtype_str]
            for dtype_str in self.data.dtypes.model_dump().values()
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
            ValueError: If the precision is not suppoerted or incorrect.
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

    def validation_metric(self) -> Tuple[str, int]:
        """This method will parse the validation metric.

        Returns:
            Tuple[str, int]:
                str: The name of the metric to use for validation.
                int: The cutoff to use for validation.
        """
        metric_name, top_k = self.general.validation_metric.split("@")
        return metric_name, int(top_k)

    def convert_params(self, model_name: str, param_dict: dict) -> dict:
        """This method will parse the parameters of a given model
        and return them in the correct format.

        Args:
            model_name (str): The name of the model to parse.
            param_dict (dict): The dictionary containing the parameters to parse.

        Returns:
            dict: The dictionary with the parsed parameters in \
                the format {param_name: hyperopt_object, ...}

        Raises:
            ValueError: If the model name is not present in the MODEL_REGISTRY.
        """
        model_class = MODEL_REGISTRY.get(model_name)
        if not model_class:
            raise ValueError(f"The model {model_name} not found in MODEL_REGISTRY.")
        model_instance = model_class()
        return model_instance.get_params(param_dict)


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
