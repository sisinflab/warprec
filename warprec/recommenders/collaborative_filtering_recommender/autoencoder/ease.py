# pylint: disable = R0801, E1102
from typing import Any

import numpy as np
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="EASE")
class EASE(ItemSimRecommender):
    """Implementation of EASE algorithm from
        Embarrassingly Shallow Autoencoders for Sparse Data 2019.

    For further details, check the `paper <https://arxiv.org/abs/1905.03375>`_.

    Args:
        params (dict): The dictionary with the model params.
        interactions (Interactions): The training interactions.
        info (dict): The dictionary containing dataset information.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        l2 (float): The normalization value.
    """

    l2: float

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, info, *args, seed=seed, **kwargs)

        X = interactions.get_sparse()

        G = X.T @ X + self.l2 * np.identity(X.shape[1])
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        self.item_similarity = B
