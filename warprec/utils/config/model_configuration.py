from typing import List, Optional, Union, ClassVar, Any, Dict
from abc import ABC

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from warprec.utils.enums import (
    SearchAlgorithms,
    Schedulers,
    SearchSpace,
)
from warprec.utils.registry import (
    model_registry,
    params_registry,
    search_algorithm_registry,
    lr_scheduler_registry,
)
from warprec.utils.config.common import _scientific_notation_conversion
from warprec.utils.logger import logger

# Accepted field formats for model parameters
LIST_INT_FIELD = Union[List[Union[str, List[int]]], List[List[int]], List[int]]
INT_FIELD = Union[List[Union[str, int]], int]
FLOAT_FIELD = Union[List[Union[str, float]], float]
STR_FIELD = Union[List[str], str]
BOOL_FIELD = Union[List[Union[str, bool]], bool]


class Meta(BaseModel):
    """Definition of the Meta-information sub-configuration of a RecommenderModel.

    Attributes:
        save_model (Optional[bool]): Whether save or not the model state after training.
        save_recs (Optional[bool]): Whether save or not the recommendations after training.
        load_from (Optional[str]): The path where a previous model state has been saved.
        low_memory (Optional[bool]): Wether or not to compute data needed for the model to
            train in lazy mode. Defaults to False.
    """

    save_model: Optional[bool] = False
    save_recs: Optional[bool] = False
    load_from: Optional[str] = None
    low_memory: Optional[bool] = False


class Properties(BaseModel):
    """Definition of the Properties of the search algorithm and
    the scheduler.

    Some of these attributes are required for Ray Tune to work properly.

    Attributes:
        mode (Optional[str]): Wether to maximize or minimize the metric/loss.
            - min: Minimize the validation metric.
            - max: Maximize the validation metric.
        desired_training_it (Optional[str]): Which strategy to use
            during validation folding to determine the number of
            iterations to use in the final evaluation.
            - median: The median of the iterations of the best folds.
            - mean: The mean of the iterations of the best folds.
            - min: The min of the iterations of the best folds.
            - max: The max of the iterations of the best folds.
        seed (Optional[int]): The seed to use during optimization.
            This parameter will make the experiment reproducible.
        time_attr (Optional[str]): The measure of time that will be used
            by the scheduler.
        max_t (Optional[int]): Max time unit given to each trial.
        grace_period (Optional[int]): Min time unit given to each trial.
        reduction_factor (Optional[float]): Halving rate of trials.
    """

    mode: Optional[str] = "max"
    desired_training_it: Optional[str] = "median"
    seed: Optional[int] = 42
    time_attr: Optional[str] = None
    max_t: Optional[int] = None
    grace_period: Optional[int] = None
    reduction_factor: Optional[float] = None

    @field_validator("mode")
    @classmethod
    def check_mode(cls, v: str):
        """Validate mode."""
        if v is None:
            raise ValueError("Mode must be provided.")
        if v.lower() not in ["min", "max"]:
            raise ValueError("Mode should be either min or max.")
        return v.lower()

    @field_validator("desired_training_it")
    @classmethod
    def check_desired_training_it(cls, v: str):
        """Validate desired_training_it."""
        if v is None:
            raise ValueError("Desired_training_it must be provided.")
        if v.lower() not in ["median", "mean", "min", "max"]:
            raise ValueError(
                "Desired_training_it should be either: median, mean, min or max."
            )
        return v.lower()


class LRScheduler(BaseModel):
    """Definition of the learning rate scheduling configuration.

    Attributes:
        name (Optional[str]): The name of the scheduler.
        params (Optional[Dict[str, Any]]): The parameters of the scheduler.
    """

    name: Optional[str]
    params: Optional[Dict[str, Any]]

    @field_validator("name")
    @classmethod
    def check_name(cls, v: str):
        """Validate name."""
        if v.upper() not in lr_scheduler_registry.list_registered():
            raise ValueError(
                "The learning rate scheduler name provided is not supported. These are the "
                f"supported strategies: {lr_scheduler_registry.list_registered()}"
            )
        return v


class Optimization(BaseModel):
    """Definition of the Optimization sub-configuration of a RecommenderModel.

    Attributes:
        strategy (Optional[SearchAlgorithms]): The strategy to use in the optimization.
            - grid: Performs grid search over all the parameters provided.
            - random: Random search over the param space.
            - hopt: Bayesian optimization using HyperOptOptimization.
            - optuna: Optuna optimization, more information can be found at:
                https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html.
            - bohb: BOHB optimization, more information can be found at:
                https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.bohb.TuneBOHB.html.
        scheduler (Optional[Schedulers]): The scheduler to use in optimization.
            - fifo: Classic First In First Out trial optimization.
            - asha: ASHA Scheduler, more information can be found at:
                https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.ASHAScheduler.html.
        lr_scheduler (Optional[LRScheduler]): The learning rate scheduling options.
        properties (Optional[Properties]): The attributes required for Ray Tune to work.
        device (Optional[str]): The device that will be used for tensor operations.
            Overrides general device.
        cpu_per_trial (Optional[int]): The number of CPU cores to assign to
            each trial. Defaults to 1.
        gpu_per_trial (Optional[float]): The number of GPU to assign to
            each trial. Defaults to 0.
        num_gpus (Optional[int]): The number of gpus to assign to each trial.
            Defaults to None. If multi_gpu is set to False, this value will be ignored.
        block_size (Optional[int]): The number of items to process during prediction.
            Used by some neural models, increasing this value will affect memory usage.
        chunk_size (Optional[int]): The size of the chunk processed during prediction.
            Used by some deep models to optimize the evaluation process.
        num_samples (Optional[int]): The number of trials that Ray Tune will try.
            In case of a grid search, this parameter should be set to 1.
        checkpoint_to_keep (Optional[int]): The number of checkpoints to keep
            in the ray directory.
    """

    strategy: Optional[SearchAlgorithms] = SearchAlgorithms.GRID
    scheduler: Optional[Schedulers] = Schedulers.FIFO
    lr_scheduler: Optional[LRScheduler] = None
    properties: Optional[Properties] = Field(default_factory=Properties)
    device: Optional[str] = None
    cpu_per_trial: Optional[int] = 1
    gpu_per_trial: Optional[float] = 0
    num_gpus: Optional[int] = None
    block_size: Optional[int] = 50
    chunk_size: Optional[int] = 4096
    num_samples: Optional[int] = 1
    checkpoint_to_keep: Optional[int] = 5

    @field_validator("strategy")
    @classmethod
    def check_strategy(cls, v: str):
        """Validate strategy."""
        if v.upper() not in search_algorithm_registry.list_registered():
            raise ValueError(
                "The strategy provided is not supported. These are the "
                f"supported strategies: {search_algorithm_registry.list_registered()}"
            )
        return v

    @field_validator("device")
    @classmethod
    def check_device(cls, v: str):
        """Validate device."""
        if v is None:
            return v

        if v in ("cuda", "cpu"):
            if v == "cuda" and not torch.cuda.is_available():
                raise ValueError(
                    "Cuda device was selected but not available on current machine."
                )
            return v
        raise ValueError(f'Device {v} is not supported. Use "cpu" or "cuda".')

    @field_validator("cpu_per_trial")
    @classmethod
    def check_cpu_per_trial(cls, v: int):
        """Validate cpu_per_trial."""
        if v <= 0:
            raise ValueError("Requested a number of CPU cores less or equal to zero.")
        return v

    @field_validator("gpu_per_trial")
    @classmethod
    def check_gpu_per_trial(cls, v: int):
        """Validate gpu_per_trial."""
        if v < 0:
            logger.attention("Found a value of 'gpu_per_trial' < 0. Defaulting to 0.")
            v = 0

        if v > 0 and not v.is_integer():
            raise ValueError(
                "Number of 'gpu_per_trial' not supported. Supported values must be "
                "in the range (0, 1] or integer values > 1, like 2, 3, ..."
            )

        return v

    @model_validator(mode="after")
    def model_validation(self):
        """Optimization model validation."""
        if self.strategy == SearchAlgorithms.GRID and self.num_samples > 1:
            logger.attention(
                f"You are running a grid search with num_samples {self.num_samples}, "
                f"this will run extra samples. Check your configuration "
                f"for possible mistakes."
            )
        if self.scheduler == Schedulers.FIFO:
            if self.properties.time_attr is not None:
                logger.attention(
                    "You have passe the field time_attribute but FIFO "
                    "scheduling does not require it."
                )
            if self.properties.max_t is not None:
                logger.attention(
                    "You have passe the field max_t but FIFO "
                    "scheduling does not require it."
                )
            if self.properties.grace_period is not None:
                logger.attention(
                    "You have passe the field grace_period but FIFO "
                    "scheduling does not require it."
                )
            if self.properties.reduction_factor is not None:
                logger.attention(
                    "You have passe the field reduction_factor but FIFO "
                    "scheduling does not require it."
                )
        if self.scheduler == Schedulers.ASHA:
            if self.properties.max_t is None:
                raise ValueError(
                    "Max_t property is required for ASHA scheduling. "
                    "Change type of scheduling or provide the max_t attribute."
                )
            if self.properties.grace_period is None:
                raise ValueError(
                    "Grace_period property is required for ASHA scheduling. "
                    "Change type of scheduling or provide the grace_period attribute."
                )
            if self.properties.reduction_factor is None:
                raise ValueError(
                    "Reduction_factor property is required for ASHA scheduling. "
                    "Change type of scheduling or provide the reduction_factor attribute."
                )

        return self


class EarlyStopping(BaseModel):
    """Definition of the Early stopping sub-configuration of a RecommenderModel.

    Attributes:
        monitor (Optional[str]): The value to monitor. Can be either 'score'
            or 'loss'.
        patience (int): The number of epochs the trainer will wait for
            an improvement.
        grace_period (Optional[int]): The number of epochs to wait before
            activating the early stopping.
        min_delta (Optional[float]): The delta value used to offset the improvement.
    """

    monitor: Optional[str] = "score"
    patience: int
    grace_period: Optional[int] = 0
    min_delta: Optional[float] = 0.0

    @field_validator("monitor")
    @classmethod
    def check_monitor(cls, v: str):
        """Validate monitor."""
        if v not in ["score", "loss"]:
            raise ValueError(
                "Value for monitor must be either 'score' or 'loss'. "
                f"Value received: {v}"
            )
        return v

    @field_validator("patience")
    @classmethod
    def check_patience(cls, v: int):
        """Validate patience."""
        if v <= 0:
            raise ValueError("Value for patience must be >0.")
        return v

    @field_validator("grace_period")
    @classmethod
    def check_grace_period(cls, v: int):
        """Validate grace_period."""
        if v < 0:
            raise ValueError("Value for grace_period must be >=0.")
        return v


class RecomModel(BaseModel, ABC):
    """Definition of a RecommendationModel configuration. All models must extend this class.

    Attributes:
        model_config: Configuration of the PyDantic model.
        meta (Meta): The meta-information about the model. Defaults to Meta default values.
        optimization (Optimization): The optimization information that will be used by Ray Tune.
        early_stopping (Optional[EarlyStopping]): The early stopping information that
            will be used during training.
        need_side_information (ClassVar[bool]): Wether or not the model needs side information.
        need_single_trial_validation (ClassVar[bool]): Wether or not the model needs to be
            validated during training.
    """

    model_config = ConfigDict(extra="allow")

    meta: Meta = Field(default_factory=Meta)
    optimization: Optimization = Field(default_factory=Optimization)
    early_stopping: Optional[EarlyStopping] = None
    need_side_information: ClassVar[bool] = False
    need_single_trial_validation: ClassVar[bool] = False

    @model_validator(mode="after")
    def model_validation(self):
        """RecomModel model validation."""
        _name = self.__class__.__name__

        # If validation has not been registered, return the model not
        # validated
        if _name.upper() not in params_registry.list_registered():
            for field, value in self.model_extra.items():
                if not isinstance(value, list):
                    self.model_extra[field] = [value]

                current_list = self.model_extra[field]

                # Clean scientific notation if present
                current_list = _scientific_notation_conversion(current_list)

                # Check if a search space has been provided
                if current_list and not any(
                    current_list[0] == ss for ss in SearchSpace
                ):
                    # If no search space has been provided, use default
                    if self.optimization.strategy == SearchAlgorithms.GRID:
                        current_list.insert(0, SearchSpace.GRID)
                    else:
                        current_list.insert(0, SearchSpace.CHOICE.value)
                self.model_extra[field] = current_list

            return self

        # Create mapping of {field: typing}
        field_to_type = {}
        for field, typing in self.__class__.__annotations__.items():
            field_to_type[field] = typing

        # Basic controls
        self.validate_model(_name)

        # General parameters control
        updated_values = self.model_dump(
            exclude=["meta", "optimization", "early_stopping"]
        )

        for field, value in updated_values.items():
            # In case of cross-validation the fold becomes an hyperparameter
            # that we do not need to handle/validate
            if field == "fold":
                continue
            typing = field_to_type[field]
            if self.optimization.strategy == SearchAlgorithms.GRID:
                updated_values[field] = self.validate_grid_search(field, value)
            else:
                updated_values[field] = self.validate_other_search(field, value, typing)

        self.__dict__.update(updated_values)
        return self

    def validate_model(self, name: str):
        """Checks if the model exist in the registry.

        Args:
            name (str): The name of the model.

        Raises:
            ValueError: If model is not registered.
        """
        if name.upper() not in model_registry.list_registered():
            raise ValueError(
                f"Model {name} not in model_registry. "
                f"These are the available models: {model_registry.list_registered()}."
            )

    def validate_grid_search(
        self, field: str, value: List[Union[str, float, int]]
    ) -> List[Union[str, float, int]]:
        """Validates Grid Search specific constraints.

        Args:
            field (str): The name of the field.
            value (List[Union[str, float, int]]): The parameter list.

        Returns:
            List[Union[str, float, int]]: The grid search validated parameter list.

        Raises:
            ValueError: If the values are not of the same type.
        """
        if isinstance(value[0], str):
            _strat: str = value[0]
            if _strat.lower() == SearchSpace.GRID.value:
                value.pop(0)
        if all(isinstance(item, type(value[0])) for item in value):
            return [SearchSpace.GRID] + value
        raise ValueError(
            f"For the Grid Search optimization, the field {field} must "
            f"have values of the same type. "
            f"Values received: {value}."
        )

    def validate_other_search(
        self, field: str, value: List[Union[str, float, int]], typing: Any
    ) -> List[Union[str, float, int]]:
        """Validates search strategies other than Grid Search.

        Args:
            field (str): The name of the field.
            value (List[Union[str, float, int]]): The parameter list.
            typing (Any): The type of the field.

        Returns:
            List[Union[str, float, int]]: The validated parameter list.

        Raises:
            ValueError: If the range is not in a correct format.
        """
        # We check if a search space has been provided
        # if yes, then we validate the strategy the user provided
        if isinstance(value[0], str) and value[0].lower() in [
            space.value for space in SearchSpace
        ]:
            value = self.check_valid_strategy(value[0], typing, value)
        else:
            # If a strategy has not been provided, we use a default one
            value.insert(0, SearchSpace.CHOICE)

        # If the typing is simple, then we don't need further checks
        if (
            typing in [STR_FIELD, BOOL_FIELD, LIST_INT_FIELD]
            or value[0] == SearchSpace.CHOICE
        ):
            return value

        # Final checks for list of values length and distribution constraint
        if len(value) < 3 or len(value) > 5:
            raise ValueError(
                f"Invalid range format for field {field}. "
                f"Expected [1.0, 5.0] or ['uniform', 1.0, 5.0]. "
                f"Received: {value}."
            )

        return self.validate_advanced_distribution(field, value)

    def check_valid_strategy(
        self,
        strat: Union[SearchSpace, str],
        typing: Any,
        value: List[Union[str, float, int]],
    ) -> List[Union[str, float, int]]:
        """Checks if the strategy provided is valid for the field type.

        Args:
            strat (Union[SearchSpace, str]): The search space strategy.
            typing (Any): The type of the field.
            value (List[Union[str, float, int]]): The parameter list.

        Returns:
            List[Union[str, float, int]]: The validated parameter list.
        """
        if typing == INT_FIELD and strat not in [
            SearchSpace.RANDINT,
            SearchSpace.QRANDINT,
            SearchSpace.LOGRANDINT,
            SearchSpace.QLOGRANDINT,
            SearchSpace.CHOICE,
        ]:
            value.pop(0)
            value.insert(0, SearchSpace.CHOICE)
            logger.attention(
                f"Strategy {strat} is not valid for field {typing}. "
                f"Choice strategy will be used instead. "
            )
        if typing == FLOAT_FIELD and strat not in [
            SearchSpace.UNIFORM,
            SearchSpace.QUNIFORM,
            SearchSpace.LOGUNIFORM,
            SearchSpace.QLOGUNIFORM,
            SearchSpace.RANDN,
            SearchSpace.QRANDN,
            SearchSpace.CHOICE,
        ]:
            value.pop(0)
            value.insert(0, SearchSpace.CHOICE)
            logger.attention(
                f"Strategy {strat} is not valid for field {typing}. "
                f"Choice strategy will be used instead. "
            )
        if typing == STR_FIELD and strat not in [
            SearchSpace.CHOICE,
        ]:
            value.pop(0)
            value.insert(0, SearchSpace.CHOICE)
            logger.attention(
                f"Strategy {strat} is not valid for field {typing}. "
                f"Choice strategy will be used instead. "
            )
        if typing == BOOL_FIELD and strat not in [
            SearchSpace.CHOICE,
        ]:
            value.pop(0)
            value.insert(0, SearchSpace.CHOICE)
            logger.attention(
                f"Strategy {strat} is not valid for field {typing}. "
                f"Choice strategy will be used instead. "
            )
        if typing == LIST_INT_FIELD and strat not in [
            SearchSpace.CHOICE,
        ]:
            value.pop(0)
            value.insert(0, SearchSpace.CHOICE)
            logger.attention(
                f"Strategy {strat} is not valid for field {typing}. "
                f"Choice strategy will be used instead. "
            )
        return value

    def validate_basic_range(
        self, field: str, value: List[Union[str, float, int]]
    ) -> List[Union[str, float, int]]:
        """Validates simple numerical ranges.

        Args:
            field (str): The name of the field.
            value (List[Union[str, float, int]]): The parameter list.

        Returns:
            List[Union[str, float, int]]: The parameter in the uniform format.

        Raises:
            ValueError: If values are not numbers.
        """
        if all(isinstance(item, (float, int)) for item in value):
            return [SearchSpace.UNIFORM.value] + [float(item) for item in value]
        raise ValueError(
            f"The range for field {field} must contain numbers. Received: {value}."
        )

    def validate_advanced_distribution(
        self, field: str, value: List[Union[str, float, int]]
    ) -> List[Union[str, float, int]]:
        """Validates complex search space distributions.

        Args:
            field (str): The name of the field.
            value (List[Union[str, float, int]]): The parameter list.

        Returns:
            List[Union[str, float, int]]: The values in uniformed format.

        Raises:
            ValueError: If the SearchSpace is not in the registry.
        """
        # Extract the main values
        strat = value[0]  # Selected SearchSpace
        f1, f2 = float(value[1]), float(value[2])  # Lower and upper bound

        # This check is shared between all search spaces
        if f2 <= f1:
            raise ValueError(
                "The upper bound must be higher than the lower bound. "
                f"Received: {value} for field {field}"
            )

        match strat:
            case SearchSpace.QUNIFORM | SearchSpace.QRANDN | SearchSpace.QRANDINT:
                # Check if the list of values is of expected length
                if len(value) != 4:
                    raise ValueError(
                        "Quantized distributions require 3 values: "
                        "the first and second value represent the "
                        "lower and upper bound. The third value is the quantization constant. "
                        "Expected value: ['quniform', 5, 100, 5]. "
                        f"Received: {value} for field {field}"
                    )
                # Check on the quantization constant
                q = float(value[3])
                if q <= 0:
                    raise ValueError(
                        "The quantization constant must be positive. "
                        f"Received: {value} for field {field}"
                    )
                if f1 % q != 0 or f2 % q != 0:
                    raise ValueError(
                        "The upper and lower bound must be divisible by the "
                        "quantization factor. "
                        f"Received: {value} for field {field}"
                    )
            case SearchSpace.LOGUNIFORM | SearchSpace.LOGRANDINT:
                # Check if the list of values is of expected length
                if len(value) not in [3, 4]:
                    raise ValueError(
                        "Logarithmic distributions require 2 or 3 values: "
                        "the first and second value are required and represent the "
                        "lower and upper bound. The third value is optional and is the base of "
                        "the logarithm. Expected value: ['loguniform', 1e-4, 1e-2] or ['loguniform', 1e-4, 1e-2, 2]. "
                        f"Received: {value} for field {field}"
                    )

                # Check for the optional log base
                f3 = int(value[3]) if len(value) == 4 else None
                if f3 is not None and f3 <= 0:
                    raise ValueError(
                        f"Logarithm base must be positive. "
                        f"Received: {value} for field {field}."
                    )
            case SearchSpace.QLOGUNIFORM | SearchSpace.QLOGRANDINT:
                # Check if the list of values is of expected length
                if len(value) not in [4, 5]:
                    raise ValueError(
                        "Quantized logarithmic distributions require 3 or 4 values: "
                        "the first and second value are required and represent the "
                        "lower and upper bound. The third value is the quantization constant. "
                        "The fourth value is optional and is the base of the logarithm."
                        "Expected value: ['qloguniform', 1e-4, 1e-2, 1e-3] or ['qloguniform', 1e-4, 1e-2, 1e-3, 2]. "
                        f"Received: {value} for field {field}"
                    )
                # Check on the quantization constant
                q = float(value[3])
                if q <= 0:
                    raise ValueError(
                        "The quantization constant must be positive. "
                        f"Received: {value} for field {field}"
                    )
                if q > f2:
                    raise ValueError(
                        "The quantization constant must be lower then "
                        "the upper bound. "
                        f"Received: {value} for field {field}"
                    )
                # Check for the optional log base
                f3 = int(value[4]) if len(value) == 5 else None
                if f3 is not None and f3 <= 0:
                    raise ValueError(
                        f"Logarithm base must be positive. "
                        f"Received: {value} for field {field}."
                    )

        return value

    def validate_single_trial_params(self):
        """This method must be implemented by models that present possible
        inconsistencies in their params.

        Ray Tune will call this method to validate a possible configuration
        of parameters.
        """

    def validate_all_combinations(self):
        """This method validates all possible combinations and ensures that
        at least one is valid.
        """

    def _clean_param_list(
        self,
        param_list: list,
    ):
        """Helper method to clean parameter lists from search space information."""
        if param_list and isinstance(param_list[0], str):
            return param_list[1:]
        return param_list
