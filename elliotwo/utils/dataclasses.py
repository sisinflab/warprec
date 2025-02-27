from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from elliotwo.evaluation.metrics import AbstractMetric


@dataclass
class TrainerConfig:
    """This class will store all the training information need
    for the hyperparameter optimization.

    Attributes:
        model_name (str): The name of the model to optimize.
        param (dict): The parameters of the model already in
            Ray Tune format.
        metric (AbstractMetric): The metric to use as validation.
        top_k (int): The cutoff tu use as validation.
    """

    model_name: str
    param: dict
    metric: "AbstractMetric"
    top_k: int
