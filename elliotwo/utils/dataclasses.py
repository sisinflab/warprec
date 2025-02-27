from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from elliotwo.evaluation.metrics import AbstractMetric
    from elliotwo.data.dataset import AbstractDataset


@dataclass
class TrainerConfig:
    """This class will store all the training information need
    for the hyperparameter optimization.

    Attributes:
        model_name (str): The name of the model to optimize.
        dataset (AbstractDataset): The dataset on which optimize the model.
        param_space (dict): The param space to optimize using Ray Tune.
        metric (AbstractMetric): The metric to use as validation.
        top_k (int): The cutoff tu use as validation.
    """

    model_name: str
    dataset: "AbstractDataset"
    param_space: dict
    metric: "AbstractMetric"
    top_k: int
