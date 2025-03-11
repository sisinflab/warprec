from typing import Any

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import ElasticNet
from elliotwo.recommenders.abstract_recommender import ItemSimilarityRecommender
from elliotwo.data.dataset import Interactions
from elliotwo.utils.registry import model_registry


@model_registry.register(name="Slim")
class Slim(ItemSimilarityRecommender):
    """Implementation of Slim model from Sparse Linear Methods for Top-N Recommender Systems 2011.

    Attributes:
        l1 (float): The normalization value.
        alpha (float): The alpha multiplication constant value.

    Args:
        params (dict): The dictionary with the model params.
        items (int): The number of items that will be learned.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments

    The params allowed by this model are as follows:
        l1 (float): Normalization parameter to use during train.
        alpha (float): Normalization parameter to use during train.
    """

    l1: float
    alpha: float

    def __init__(self, params: dict, items: int, *args: Any, **kwargs: Any):
        super().__init__(params, items, *args, **kwargs)
        self._name = "Slim"

    def fit(self, interactions: Interactions, *args: Any, **kwargs: Any):
        """During training we will compute the B similarity matrix {item x item}."""
        # Predefine the number of items, similarity matrix and ElasticNet
        X = interactions.get_sparse()

        num_items = X.shape[1]
        item_sim = np.zeros((num_items, num_items))

        for i in range(num_items):
            # x_i represent the item column from training set
            x_i = X[:, i].toarray().ravel()

            # X_j will contain all the other columns
            mask = np.arange(num_items) != i
            X_j = X[:, mask]

            # Use ElasticNet as in the paper
            model = ElasticNet(
                alpha=self.alpha, l1_ratio=self.l1, fit_intercept=False, positive=True
            )
            model.fit(X_j, x_i)

            # Get coefficients and use them in the similarity matrix
            coef = model.coef_
            item_sim[i, mask] = coef
        self.item_similarity = nn.Parameter(torch.from_numpy(item_sim))
