# pylint: disable = R0801, E1102
from typing import Any

import numpy as np
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="AddEASE")
class AddEASE(ItemSimRecommender):
    """Implementation of AddEASE algorithm from
        Closed-Form Models for Collaborative Filtering with Side-Information 2020.

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/3383313.3418480>`_.

    Args:
        params (dict): The dictionary with the model params.
        interactions (Interactions): The training interactions.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        l2 (float): The normalization value.
        alpha (float): The alpha constant value.
    """

    l2: float
    alpha: float

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, seed=seed, info=info, *args, **kwargs)

        X = interactions.get_sparse()
        item_profile = interactions.get_side_sparse()

        # Fist solution
        G = X.T @ X + self.l2 * np.identity(X.shape[1])
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        # Second solution
        P = item_profile @ item_profile.T + self.l2 * np.identity(item_profile.shape[0])
        U = np.linalg.inv(P)
        U /= -np.diag(U)
        np.fill_diagonal(U, 0.0)

        # Linear combination
        self.item_similarity = self.alpha * B + (1 - self.alpha) * U
