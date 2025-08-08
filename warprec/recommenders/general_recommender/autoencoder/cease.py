# pylint: disable = R0801, E1102
from typing import Any

import numpy as np
import torch
from torch import nn
from scipy.sparse import vstack
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.data.dataset import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="CEASE")
class CEASE(ItemSimRecommender):
    """Implementation of CEASE algorithm from
        Closed-Form Models for Collaborative Filtering with Side-Information 2020.

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/3383313.3418480>`_.

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
        alpha (float): The alpha constant value.
    """

    l2: float
    alpha: float

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
        self._name = "CEASE"

        X = interactions.get_sparse()
        item_profile = interactions.get_side_sparse()

        # Extend interaction matrix
        X_extended = vstack((X, item_profile.T * self.alpha))

        G = X_extended.T @ X_extended + self.l2 * np.identity(X_extended.shape[1])
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        self.item_similarity = nn.Parameter(torch.tensor(B, dtype=torch.float32))
