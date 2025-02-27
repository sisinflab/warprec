# pylint: disable=E1101
from typing import List, Optional, Union
from abc import ABC

from pydantic import BaseModel, Field, model_validator, field_validator
from elliotwo.utils.registry import params_registry, model_registry, metric_registry


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


class Optimization(BaseModel):
    """Definition of the Optimization sub-configuration of a RecommenderModel.

    Attributes:
        strategy (Optional[str]): The strategy to use in the optimization.
            - grid: Performs grid search over all the parameters provided.
            - hopt: Bayesian optimization using HyperOptOptimization.
        validation_metric (Optional[str]): The metric/loss that will validate each trial in Ray Tune.
        mode (Optional[str]): Wether to maximize or minimize the metric/loss.
            - min: Minimize the validation metric.
            - max: Maximize the validation metric.
        num_samples (Optional[int]): The number of trials that Ray Tune will try.
            In case of a grid search, this parameter should be set to 1.
        cpu_per_trial (Optional[float]): The number of cpu cores dedicated to
            each trial.
        gpu_per_trial (Optional[float]): The number of gpu dedicated to
            each trial.
    """

    strategy: Optional[str] = "grid"
    validation_metric: Optional[str] = "NDCG@10"
    mode: Optional[str] = "max"
    num_samples: Optional[int] = 1
    cpu_per_trial: Optional[float] = 1.0
    gpu_per_trial: Optional[float] = 0.0

    @field_validator("strategy")
    @classmethod
    def check_strategy(cls, v):
        """Validate strategy."""
        supported_strategies = ["grid", "hopt"]
        if v not in supported_strategies:
            raise ValueError(
                "The strategy provided is not supported. These are the "
                f"supported strategies: {supported_strategies}"
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

    @field_validator("mode")
    @classmethod
    def check_mode(cls, v):
        """Validate mode."""
        if v not in ["min", "max"]:
            raise ValueError("Mode should be either min or max.")
        return v


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
        _range_strat = ["hopt"]
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
                updated_values[field] = [value]
            if self.optimization.strategy in _range_strat and len(value) > 2:
                raise ValueError(
                    f"For the strategy {self.optimization.strategy} values of {field} are "
                    f"expected to range like [1.0, 5.0]. Value received {value}"
                )

        self.__dict__.update(updated_values)
        return self


@params_registry.register("EASE")
class EASE(RecomModel):
    """Definition of the model EASE.

    Attributes:
        l2 (Union[List[float], float]): List of values that l2 regularization can take.
    """

    l2: Union[List[float], float]


@params_registry.register("Slim")
class Slim(RecomModel):
    """Definition of the model Slim.

    Attributes:
        l1 (Union[List[float], float]): List of values that l1 regularization can take.
        alpha (Union[List[float], float]): List of values that alpha can take.
    """

    l1: Union[List[float], float]
    alpha: Union[List[float], float]
