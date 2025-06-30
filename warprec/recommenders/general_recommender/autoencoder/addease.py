# pylint: disable = R0801, E1102
from typing import Optional, Callable, Any

import numpy as np
import torch
from torch import nn
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.data.dataset import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="AddEASE")
class AddEASE(ItemSimRecommender):
    """Implementation of AddEASE algorithm from
        Closed-Form Models for Collaborative Filtering with Side-Information 2020.

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/3383313.3418480>`_.

    Args:
        params (dict): The dictionary with the model params.
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
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, info=info, *args, **kwargs)
        self._name = "AddEASE"

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        The training will be conducted on the sparse representation of the interactions
        and the sparse representation of the features.
        During the train a similarity matrix {item x item} will be learned.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        # The classic implementation follows the original paper
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
        sim = self.alpha * B + (1 - self.alpha) * U

        self.item_similarity = nn.Parameter(torch.tensor(sim, dtype=torch.float32))

        if report_fn is not None:
            report_fn(self)
