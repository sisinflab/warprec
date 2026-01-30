# pylint: disable = R0801, E1102
from typing import Any

import scipy.sparse as sp
from sklearn.linear_model import ElasticNet
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="Slim")
class Slim(ItemSimRecommender):
    """Implementation of Slim model from
        Sparse Linear Methods for Top-N Recommender Systems 2011.

    For further details, check the `paper <https://ieeexplore.ieee.org/document/6137254>`_.

    Args:
        params (dict): The dictionary with the model params.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        l1 (float): The normalization value.
        alpha (float): The alpha multiplication constant value.
    """

    l1: float
    alpha: float

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

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

        # Stack the coefficients and make the matrix dense
        self.item_similarity = sp.vstack(item_coeffs).T.todense()
