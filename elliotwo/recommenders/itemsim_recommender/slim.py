# pylint: disable=invalid-name
import numpy as np
from sklearn.linear_model import ElasticNet
from elliotwo.data.dataset import AbstractDataset
from elliotwo.utils.config import Configuration
from elliotwo.recommenders.abstract_recommender import ItemSimilarityRecommender
from elliotwo.utils.registry import model_registry


@model_registry.register(name="Slim")
class Slim(ItemSimilarityRecommender):
    """Implementation of Slim model from Sparse Linear Methods for Top-N Recommender Systems 2011.

    Args:
        config (Configuration): The configuration of the experiement.
        dataset (AbstractDataset): The dataset to train the model on.
        params (dict): The parameters of the model.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments

    The params allowed by this model are as follows:
        l1 (float): Normalization parameter to use during train.
        alpha (float): Normalization parameter to use during train.
    """

    def __init__(
        self,
        config: Configuration,
        dataset: AbstractDataset,
        params: dict,
        *args,
        **kwargs,
    ):
        super().__init__(config, dataset, params, *args, **kwargs)

        self._name = "Slim"
        self.X = self.interaction_matrix
        self.l1 = self._params["l1"]
        self.alpha = self._params["alpha"]

    def fit(self):
        """During training we will compute the B similarity matrix {item x item}."""
        # Predefine the number of items, similarity matrix and ElasticNet
        num_items = self.X.shape[1]
        self.item_similarity = np.zeros((num_items, num_items))

        for i in range(num_items):
            # x_i represent the item column from training set
            x_i = self.X[:, i].toarray().ravel()

            # X_j will contain all the other columns
            mask = np.arange(num_items) != i
            X_j = self.X[:, mask]

            # Use ElasticNet as in the paper
            model = ElasticNet(
                alpha=self.alpha, l1_ratio=self.l1, fit_intercept=False, positive=True
            )
            model.fit(X_j, x_i)

            # Get coefficients and use them in the similarity matrix
            coef = model.coef_
            self.item_similarity[i, mask] = coef
