from typing import List, Optional, Union
from abc import ABC

from pydantic import BaseModel, Field, model_validator, field_validator
from elliotwo.utils.enums import (
    SearchAlgorithms,
    Schedulers,
    SearchSpace,
)
from elliotwo.utils.registry import (
    params_registry,
    model_registry,
    metric_registry,
    search_algorithm_registry,
    search_space_registry,
)
from elliotwo.utils.logger import logger


class Meta(BaseModel):
    """Definition of the Meta-information sub-configuration of a RecommenderModel.

    Attributes:
        save_model (Optional[bool]): Whether save or not the model state after training.
        load_from (Optional[str]): The path where a previous model state has been saved.
        implementation (Optional[str]): The implementation to be used.
    """

    save_model: Optional[bool] = False
    load_from: Optional[str] = None
    implementation: Optional[str] = "latest"


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

    mode: Optional[str] = None
    seed: Optional[int] = None
    time_attr: Optional[str] = None
    max_t: Optional[int] = None
    grace_period: Optional[int] = None
    reduction_factor: Optional[float] = None

    @field_validator("mode")
    @classmethod
    def check_mode(cls, v: str):
        """Validate mode."""
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
            - optuna: Optuna optimization, more information can
                be found at https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html.
            - bohb: BOHB optimization, more information can
                be found at https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.bohb.TuneBOHB.html.
        scheduler (Optional[Schedulers]): The scheduler to use in optimization.
            - fifo: Classic First In First Out trial optimization.
            - asha: ASHA Scheduler, more information can be found
                at https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.ASHAScheduler.html.
        properties (Optional[Properties]): The attributes required for Ray Tune to work.
        validation_metric (Optional[str]): The metric/loss that will validate each trial in Ray Tune.
        num_samples (Optional[int]): The number of trials that Ray Tune will try.
            In case of a grid search, this parameter should be set to 1.
        cpu_per_trial (Optional[float]): The number of cpu cores dedicated to
            each trial.
        gpu_per_trial (Optional[float]): The number of gpu dedicated to
            each trial.
    """

    strategy: Optional[SearchAlgorithms] = SearchAlgorithms.GRID
    scheduler: Optional[Schedulers] = Schedulers.FIFO
    properties: Optional[Properties] = None
    validation_metric: Optional[str] = "NDCG@10"
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
        if metric not in metric_registry.list_registered():
            raise ValueError(
                f"Metric {metric} not in metric registry. This is the list"
                f"of supported metrics: {metric_registry.list_registered()}"
            )
        if not top_k.isnumeric():
            raise ValueError(
                "Validation metric should be provided with a top_k number."
            )
        return v

    @model_validator(mode="after")
    def model_validation(self):
        """Optimization model validation."""
        if self.strategy == SearchAlgorithms.GRID and self.num_samples > 1:
            logger.attention(
                f"You are running a grid search with num_samples {self.num_samples}, "
                f"check your configuration for possible mistakes."
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

        # Basic controls
        self.validate_model_and_implementation(_name, _imp)

        # General parameters control
        updated_values = self.model_dump(exclude=["meta", "optimization"])
        updated_values = self.normalize_values(updated_values)

        for field, value in updated_values.items():
            if self.optimization.strategy == SearchAlgorithms.GRID:
                updated_values[field] = self.validate_grid_search(field, value)
            else:
                updated_values[field] = self.validate_other_search(field, value)

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
        if name not in model_registry.list_registered():
            raise ValueError(f"Model {name} not in model_registry.")
        if imp not in model_registry.list_implementations(name):
            raise ValueError(f"Model {name} does not have {imp} implementation.")

    def normalize_values(self, values: dict) -> dict:
        """Ensures all values are lists.

        Args:
            values (dict): The dictionary with all the parameters
                of the model.

        Returns:
            dict: The dictionary of all normalized parameters.
        """
        for field, value in values.items():
            if not isinstance(value, list):
                values[field] = [value]
        return values

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
        if all(isinstance(item, type(value[0])) for item in value):
            return [SearchSpace.GRID] + value
        raise ValueError(
            f"For the Grid Search optimization, the field {field} must have values of the same type. "
            f"Values received: {value}."
        )

    def validate_other_search(
        self, field: str, value: List[Union[str, float, int]]
    ) -> List[Union[str, float, int]]:
        """Validates search strategies other than Grid Search.

        Args:
            field (str): The name of the field.
            value (List[Union[str, float, int]]): The parameter list.

        Returns:
            List[Union[str, float, int]]: The validated parameter list.

        Raises:
            ValueError: If the range is not in a correct format.
        """
        if all(isinstance(item, str) for item in value):
            return [SearchSpace.CHOICE] + value

        if len(value) < 2 or len(value) > 4:
            raise ValueError(
                f"Invalid range format for field {field}. Expected [1.0, 5.0] or ['uniform', 1.0, 5.0]. "
                f"Received: {value}."
            )

        if len(value) == 2:
            return self.validate_basic_range(field, value)

        return self.validate_advanced_distribution(field, value)

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
            return [str(SearchSpace.UNIFORM)] + [float(item) for item in value]
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
                f"{search_space} does not require a rounding factor, but extra values were provided. "
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


@params_registry.register("EASE")
class EASE(RecomModel):
    """Definition of the model EASE.

    Attributes:
        l2 (Union[List[Union[str, float, int]], float, int]): List of values that l2 regularization can take.
    """

    l2: Union[List[Union[str, float, int]], float, int]


@params_registry.register("Slim")
class Slim(RecomModel):
    """Definition of the model Slim.

    Attributes:
        l1 (Union[List[Union[str, float, int]], float, int]): List of values that l1 regularization can take.
        alpha (Union[List[Union[str, float, int]], float, int]): List of values that alpha can take.
    """

    l1: Union[List[Union[str, float, int]], float, int]
    alpha: Union[List[Union[str, float, int]], float, int]
