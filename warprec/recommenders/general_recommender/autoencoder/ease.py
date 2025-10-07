# pylint: disable = R0801, E1102
from typing import Any

import numpy as np
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.data.dataset import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="EASE")
class EASE(ItemSimRecommender):
    """Implementation of EASE algorithm from
        Embarrassingly Shallow Autoencoders for Sparse Data 2019.

    For further details, check the `paper <https://arxiv.org/abs/1905.03375>`_.

    Args:
        params (dict): The dictionary with the model params.
        interactions (Interactions): The training interactions.
        *args (Any): Argument for PyTorch nn.Module.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        l2 (float): The normalization value.
    """

    l2: float

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(
            params, interactions, device=device, seed=seed, info=info, *args, **kwargs
        )

        X = interactions.get_sparse()

        G = X.T @ X + self.l2 * np.identity(X.shape[1])
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        self.item_similarity = B
