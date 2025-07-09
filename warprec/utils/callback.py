from typing import Any, Optional, TYPE_CHECKING

from pandas import DataFrame
from ray.tune import Callback

if TYPE_CHECKING:
    from warprec.data.dataset import Dataset
    from warprec.recommenders.base_recommender import Recommender


class WarpRecCallback(Callback):
    """A base class for WarpRec callbacks.

    This class extends the Ray Tune Callback class to provide a base for custom WarpRec callbacks.
    Custom callbacks should inherit from this class and implement the necessary methods.

    Args:
        *args (Any): Additional positional arguments.
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def on_dataset_creation(
        self,
        dataset: "Dataset",
        train_set: DataFrame,
        test_set: Optional[DataFrame],
        val_set: Optional[DataFrame],
        *args: Any,
        **kwargs: Any,
    ):
        """Callback method to be called after dataset creation.

        This method can be overridden in custom callbacks to
        perform actions after the dataset is created.

        Args:
            dataset (Dataset): The dataset that has been created.
            train_set (DataFrame): The training set.
            test_set (Optional[DataFrame]): The test set.
            val_set (Optional[DataFrame]): The validation set.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_training_complete(self, model: "Recommender", *args: Any, **kwargs: Any):
        """Callback method to be called after training is complete.

        This method can be overridden in custom callbacks to
        perform actions after the training is complete.

        Args:
            model (Recommender): The trained model.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_evaluation_complete(
        self,
        model: "Recommender",
        params: dict,
        results: dict,
        *args: Any,
        **kwargs: Any,
    ):
        """Callback method to be called after model evaluation.

        This method can be overridden in custom callbacks to
        perform actions after the model evaluation is complete.

        Args:
            model (Recommender): The model that has been evaluated.
            params (dict): The parameters of the model.
            results (dict): The results of the evaluation.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """
