from typing import List, Optional, Union, Any
from abc import ABC

import torch
from pydantic import BaseModel, Field, model_validator, field_validator
from elliotwo.utils.enums import (
    SearchAlgorithms,
    Schedulers,
    SearchSpace,
)
from elliotwo.utils.registry import (
    model_registry,
    metric_registry,
    search_algorithm_registry,
    search_space_registry,
)
from elliotwo.utils.logger import logger

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
        keep_all_ray_checkpoints (Optional[bool]): Wether or not to save all the
            checkpoints for the model optimization.
        load_from (Optional[str]): The path where a previous model state has been saved.
        implementation (Optional[str]): The implementation to be used.
    """

    save_model: Optional[bool] = False
    keep_all_ray_checkpoints: Optional[bool] = False
    load_from: Optional[str] = None
    implementation: Optional[str] = "latest"

    @model_validator(mode="after")
    def model_validation(self):
        """Meta model validation."""
        if not self.save_model and self.keep_all_ray_checkpoints:
            raise ValueError(
                "You have set save_model to False but keep_all_ray_checkpoints to True. "
                "You cannot save all checkpoints if the save_model parameter has not been set."
            )
        return self


class Properties(BaseModel):
    """Definition of the Properties of the search algorithm and
    the scheduler.

    Some of these attributes are required for Ray Tune to work properly.

    Attributes:
        mode (Optional[str]): Wether to maximize or minimize the metric/loss.
            - min: Minimize the validation metric.
            - max: Maximize the validation metric.
        seed (Optional[int]): The seed to use during optimization.
            This parameter will make the experiment reproducible.
        time_attr (Optional[str]): The measure of time that will be used
            by the scheduler.
        max_t (Optional[int]): Max time unit given to each trial.
        grace_period (Optional[int]): Min time unit given to each trial.
        reduction_factor (Optional[float]): Halving rate of trials.
    """

    mode: Optional[str] = "max"
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
        properties (Optional[Properties]): The attributes required for Ray Tune to work.
        validation_metric (Optional[str]): The metric/loss that will
            validate each trial in Ray Tune.
        device (Optional[str]): The device that will be used for tensor operations.
        num_samples (Optional[int]): The number of trials that Ray Tune will try.
            In case of a grid search, this parameter should be set to 1.
        cpu_per_trial (Optional[float]): The number of cpu cores dedicated to
            each trial.
        gpu_per_trial (Optional[float]): The number of gpu dedicated to
            each trial.
    """

    strategy: Optional[SearchAlgorithms] = SearchAlgorithms.GRID
    scheduler: Optional[Schedulers] = Schedulers.FIFO
    properties: Optional[Properties] = Field(default_factory=Properties)
    validation_metric: Optional[str] = None
    device: Optional[str] = "cpu"
    num_samples: Optional[int] = 1
    cpu_per_trial: Optional[float] = 1.0
    gpu_per_trial: Optional[float] = 0.0

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

    @field_validator("validation_metric")
    @classmethod
    def check_validation_metric(cls, v: str):
        """Validate validation metric."""
        if v is None:
            raise ValueError("Validation metric must be provided.")
        if "@" not in v:
            raise ValueError(
                f"Validation metric {v} not valid. Validation metric "
                f"should be defined as: metric_name@top_k."
            )
        if v.count("@") > 1:
            raise ValueError(
                "Validation metric contains more than one @, check your configuration file."
            )
        metric, top_k = v.split("@")
        if metric.upper() not in metric_registry.list_registered():
            raise ValueError(
                f"Metric {metric} not in metric registry. This is the list"
                f"of supported metrics: {metric_registry.list_registered()}"
            )
        if not top_k.isnumeric():
            raise ValueError(
                "Validation metric should be provided with a top_k number."
            )
        return v

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
        if v.startswith("cuda:"):
            parts = v.split(":")
            if len(parts) == 2 and parts[1].isdigit():
                if parts[1] not in list(range(torch.cuda.device_count())):
                    raise ValueError(
                        f"The GPU with the idx {parts[1]} is not available. "
                        f"This is a list of available GPU idxs: "
                        f"{list(range(torch.cuda.device_count()))}."
                    )
                return v
        raise ValueError(f'Device {v} is not supported. Use "cpu" or "cuda[:index]".')

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


class RecomModel(BaseModel, ABC):
    """Definition of a RecommendationModel configuration. All models must extend this class.

    Attributes:
        meta (Meta): The meta-information about the model. Defaults to Meta default values.
        optimization (Optimization): The optimization information that will be used by Ray Tune.
    """

    meta: Meta = Field(default_factory=Meta)
    optimization: Optimization = Field(default_factory=Optimization)

    @model_validator(mode="after")
    def model_validation(self):
        """RecomModel model validation."""
        _name = self.__class__.__name__
        _imp = self.meta.implementation

        # Create mapping of {field: typing}
        field_to_type = {}
        for field, typing in self.__class__.__annotations__.items():
            field_to_type[field] = typing

        # Basic controls
        self.validate_model_and_implementation(_name, _imp)

        # General parameters control
        updated_values = self.model_dump(exclude=["meta", "optimization"])

        for field, value in updated_values.items():
            typing = field_to_type[field]
            if self.optimization.strategy == SearchAlgorithms.GRID:
                updated_values[field] = self.validate_grid_search(field, value)
            else:
                updated_values[field] = self.validate_other_search(field, value, typing)

        self.__dict__.update(updated_values)
        return self

    def validate_model_and_implementation(self, name: str, imp: str):
        """Checks if the model and its implementation exist in the registry.

        Args:
            name (str): The name of the model.
            imp (str): The name of the implementation.

        Raises:
            ValueError: If model or implementation is not registered.
        """
        if name.upper() not in model_registry.list_registered():
            raise ValueError(
                f"Model {name} not in model_registry. "
                f"These are the available models: {model_registry.list_registered()}."
            )
        if imp not in model_registry.list_implementations(name.upper()):
            raise ValueError(
                f"Model {name} does not have {imp} implementation. "
                f"These are the available implementations: "
                f"{model_registry.list_implementations(name.upper())}."
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
        # This mapping will be used to check the typing based on
        # the field typing
        choice_strat_map = {
            INT_FIELD: int,
            STR_FIELD: str,
            BOOL_FIELD: bool,
            LIST_INT_FIELD: list,
        }
        # We check if a search space has been provided
        # if yes, then we temporary remove it
        _strat = None
        if isinstance(value[0], str) and value[0].lower() in [
            space.value for space in SearchSpace
        ]:
            _strat = value.pop(0)

        # We check that values of the choice field are all of the same type
        if typing in choice_strat_map.keys():
            if all(isinstance(item, choice_strat_map[typing]) for item in value):
                if (
                    _strat
                    and isinstance(_strat, str)
                    and _strat.lower() != SearchSpace.CHOICE.value
                ):
                    logger.attention(
                        f"A different strategy from choice has been provided for a {typing}. "
                        f"The strategy has been set to choice."
                    )
                return [SearchSpace.CHOICE] + value
            else:
                raise ValueError(
                    f"{typing} expect the values to be all {str(choice_strat_map[typing])}. "
                    f"Values received: {value}"
                )

        if typing in [FLOAT_FIELD]:
            if _strat:
                value.insert(0, _strat)
            if (len(value) < 2 or len(value) > 4) and typing is not INT_FIELD:
                raise ValueError(
                    f"Invalid range format for field {field}. "
                    f"Expected [1.0, 5.0] or ['uniform', 1.0, 5.0]. "
                    f"Received: {value}."
                )

            if len(value) == 2:
                return self.validate_basic_range(field, value)

            return self.validate_advanced_distribution(field, value)

        raise ValueError("Something went wrong during model field validation.")

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
        if not isinstance(value[0], str):
            raise ValueError(
                f"Expected the first element of the parameter list to be "
                f"a valid SearchSpace. Value received {value[0]}, "
                f"SearchSpace supported {search_space_registry.list_registered()} "
            )
        _selected_search_space: str = value[0]
        _int_search_spaces = [
            SearchSpace.RANDINT,
            SearchSpace.QRANDINT,
            SearchSpace.LOGRANDINT,
            SearchSpace.QLOGRANDINT,
        ]

        if (
            _selected_search_space.upper()
            not in search_space_registry.list_registered()
        ):
            raise ValueError(
                f"{_selected_search_space} not found in SearchSpace registry. "
                f"Available options: {search_space_registry.list_registered()}."
            )

        self.check_distribution_constraints(field, value, _selected_search_space)

        if _selected_search_space not in _int_search_spaces:
            return [_selected_search_space] + [float(item) for item in value[1:]]

        return [_selected_search_space] + [int(item) for item in value[1:]]

    def check_distribution_constraints(
        self, field: str, value: List[Union[str, float, int]], search_space: str
    ):
        """Ensures that specific distributions meet their constraints.

        Args:
            field (str): The name of the field.
            value (List[Union[str, float, int]]): The parameter list.
            search_space (str): The search space selected for the field.

        Raises:
            ValueError: If the values provided do not respect the SearchSpace format.
        """
        _rounded_search_spaces = [
            SearchSpace.QUNIFORM,
            SearchSpace.QLOGUNIFORM,
            SearchSpace.QRANDN,
            SearchSpace.QRANDINT,
            SearchSpace.QLOGRANDINT,
        ]
        _log_search_spaces = [
            SearchSpace.LOGUNIFORM,
            SearchSpace.QLOGUNIFORM,
            SearchSpace.LOGRANDINT,
            SearchSpace.QLOGRANDINT,
        ]
        if search_space in _rounded_search_spaces and len(value) != 4:
            raise ValueError(
                f"{search_space} requires a rounding factor, but none was provided. "
                f"Received: {value} for field {field}."
            )
        if search_space not in _rounded_search_spaces and len(value) == 4:
            raise ValueError(
                f"{search_space} does not require a rounding factor, "
                f"but extra values were provided. "
                f"Received: {value} for field {field}."
            )
        if search_space in _rounded_search_spaces:
            i1, i2, i3 = int(value[1]), int(value[2]), int(value[3])
            if i1 % i3 != 0 or i2 % i3 != 0:
                raise ValueError(
                    f"Rounded distributions require values divisible by the round term. "
                    f"Received: {value} for field {field}."
                )
        if search_space in _log_search_spaces:
            f1, f2 = float(value[1]), float(value[2])
            if f1 <= 0 or f2 <= 0:
                raise ValueError(
                    f"Logarithmic distributions require positive values. "
                    f"Received: {value} for field {field}."
                )
