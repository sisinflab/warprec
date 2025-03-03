from typing import List, Optional, Union
from abc import ABC

from pydantic import BaseModel, Field, model_validator, field_validator
from elliotwo.utils.enums import (
    SearchAlgorithms,
    Schedulers,
    SearchSpace,
    rounded_search_spaces,
)
from elliotwo.utils.registry import (
    params_registry,
    model_registry,
    metric_registry,
    search_algorithm_registry,
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
        # This is a list of all strategies that expect data to be
        # a range format
        _name = self.__class__.__name__
        _imp = self.meta.implementation

        # Basic controls
        if _name not in model_registry.list_registered():
            raise ValueError(f"Model {_name} not in model_registry.")
        if _imp not in model_registry.list_implementations(_name):
            raise ValueError(f"Model {_name} does not have {_imp} implementation.")

        # General parameters control
        updated_values = self.model_dump(exclude=["meta", "optimization"])
        for field, value in updated_values.items():
            if not isinstance(value, list):
                value = [value]
                updated_values[field] = value

            if self.optimization.strategy == SearchAlgorithms.GRID:
                if all(isinstance(item, value[0]) for item in value):
                    updated_values[field] = [SearchSpace.GRID] + value
                else:
                    raise ValueError(
                        f"For the Grid Search optimization the field {field} "
                        f"must be all of the same type. Values received {value}."
                    )
            else:
                if all(isinstance(item, str) for item in value):
                    updated_values[field] = [SearchSpace.CHOICE] + value
                else:
                    if len(value) < 2 or len(value) > 4:
                        raise ValueError(
                            f"The value list is expected to be either: a range list, like "
                            f"[1.0, 5.0] or a range list with the first element as the "
                            f"distribution to use for the sampling process, like ['uniform', 1.0, 5.0]. "
                            f"In case of rounded distribution a fourth value is required. "
                            f"For the field {field} the value received was {value}."
                        )
                    elif len(value) == 2:
                        if all(isinstance(item, (float, int)) for item in value):
                            updated_values[field] = [SearchSpace.UNIFORM] + value
                        else:
                            raise ValueError(
                                f"The range must be in float format. For the field {field} "
                                f"the value received was {value}"
                            )
                    else:
                        if isinstance(value[0], str) and all(
                            isinstance(item, (float, int)) for item in value[1:]
                        ):
                            if (
                                value[0] in rounded_search_spaces()
                                and len(value) == 4
                                or value[0] not in rounded_search_spaces()
                                and len(value) == 3
                            ):
                                updated_values[field] = value
                            else:
                                raise ValueError(
                                    f"Rounded distribution expect three values, with the third one being "
                                    f"the approximation. Not rounded distribution expect two values. "
                                    f"You passed {value} for the field {field}."
                                )
                        else:
                            raise ValueError(
                                f"Values are expected to be all float numbers, like [1.0, 5.0] or "
                                f"they you can pass the format of the parameter space in form of a "
                                f"string as the first element in the list, like ['uniform', 1.0, 5.0]. "
                                f"You passed {value} for the field {field}."
                            )
        self.__dict__.update(updated_values)
        return self


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
