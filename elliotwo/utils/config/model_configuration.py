# pylint: disable=E1101
from typing import List, Optional, Union
from abc import abstractmethod, ABC

from ray import tune
from pydantic import BaseModel, Field, model_validator
from elliotwo.utils.registry import params_registry, model_registry


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
        validation_metric (Optional[str]): The metric/loss that will validate each trial in Ray Tune.
        mode (Optional[str]): Wether to maximize or minimize the metric/loss.
    """

    strategy: Optional[str] = "grid"
    validation_metric: Optional[str] = "NDCG@10"
    mode: Optional[str] = "max"


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

        self.__dict__.update(updated_values)
        return self

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


@params_registry.register("EASE")
class EASE(RecomModel):
    """Definition of the model EASE.

    Attributes:
        l2 (Union[List[float], float]): List of values that l2 regularization can take.
    """

    l2: Union[List[float], float]

    def get_params(self, param_dict: dict) -> dict:
        return {"l2": tune.uniform(param_dict["l2"][0], param_dict["l2"][1])}


@params_registry.register("Slim")
class Slim(RecomModel):
    """Definition of the model Slim.

    Attributes:
        l1 (Union[List[float], float]): List of values that l1 regularization can take.
        alpha (Union[List[float], float]): List of values that alpha can take.
    """

    l1: Union[List[float], float]
    alpha: Union[List[float], float]

    def get_params(self, param_dict: dict) -> dict:
        return {
            "l1": tune.uniform(param_dict["l1"][0], param_dict["l1"][1]),
            "alpha": tune.uniform(param_dict["alpha"][0], param_dict["alpha"][1]),
        }
