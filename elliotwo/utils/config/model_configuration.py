# pylint: disable=E1101
from typing import List, Optional
from abc import abstractmethod, ABC

from ray import tune
from pydantic import BaseModel, Field, field_validator, model_validator
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


class RecomModel(BaseModel, ABC):
    """Definition of a RecommendationModel configuration. All models must extend this class.

    Attributes:
        meta (Meta): The meta-information about the model. Defaults to Meta default values.
    """

    meta: Meta = Field(default_factory=Meta)

    @model_validator(mode="after")
    def model_validation(self):
        _name = self.__class__.__name__
        _imp = self.meta.implementation
        if _name not in model_registry.list_registered():
            raise ValueError(f"Model {_name} not in model_registry.")
        if _imp not in model_registry.list_implementations(_name):
            raise ValueError(f"Model {_name} does not have {_imp} implementation.")
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
        l2 (Optional[List[float]]): List of values that l2 regularization can take.
    """

    l2: Optional[List[float]] = [1.0, 2.0]

    @field_validator("l2")
    @classmethod
    def check_l2(cls, v):
        """Validates the l2 regularization.

        Raise:
            ValueError: If the l2 is not a range compatible with ray tune formulation.
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
        return {"l2": tune.uniform(param_dict["l2"][0], param_dict["l2"][1])}


@params_registry.register("Slim")
class Slim(RecomModel):
    """Definition of the model Slim.

    Attributes:
        l1 (Optional[List[float]]): List of values that l1 regularization can take.
        alpha (Optional[List[float]]): List of values that alpha can take.
    """

    l1: Optional[List[float]]
    alpha: Optional[List[float]]

    def get_params(self, param_dict: dict) -> dict:
        return {
            "l1": tune.uniform(param_dict["l1"][0], param_dict["l1"][1]),
            "alpha": tune.uniform(param_dict["alpha"][0], param_dict["alpha"][1]),
        }
