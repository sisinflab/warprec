from typing import Any

import torch
import scipy.sparse as sp
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
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    The params allowed by this model are as follows:
        l1 (float): Normalization parameter to use during train.
        alpha (float): Normalization parameter to use during train.
    """

    l1: float
    alpha: float

    def __init__(self, params: dict, info: dict, *args: Any, **kwargs: Any):
        super().__init__(params, info, *args, **kwargs)
        self._name = "Slim"

    def fit(self, interactions: Interactions, *args: Any, **kwargs: Any):
        """During training we will compute the B similarity matrix {item x item}."""
        # Predefine the number of items, similarity matrix and ElasticNet
        X = interactions.get_sparse()
        X = X.tolil()

        num_items = X.shape[1]
        item_coeffs = []
        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1,
            positive=True,
            fit_intercept=False,
            copy_X=False,
            precompute=True,
            selection="random",
            max_iter=100,
            tol=1e-4,
        )

        for j in range(num_items):
            # Current column
            r = X[:, j]

            # ElasticNet fitting
            model.fit(X, r.todense().getA1())

            # Get coefficients in sparse format
            coeffs = model.sparse_coef_

            # Add them to list
            item_coeffs.append(coeffs)

        # Stack the coefficients, make the matrix dense and
        # convert to tensor
        self.item_similarity = nn.Parameter(
            torch.from_numpy(sp.vstack(item_coeffs).T.todense())
        ).to(self._device)
