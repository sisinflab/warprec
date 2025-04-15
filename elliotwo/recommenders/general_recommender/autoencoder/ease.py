# pylint: disable = R0801, E1102
from typing import Optional, Callable, Any

import numpy as np
import torch
from torch import nn
from sklearn.utils.extmath import safe_sparse_dot
from elliotwo.recommenders.base_recommender import ItemSimRecommender
from elliotwo.data.dataset import Interactions
from elliotwo.utils.registry import model_registry


class EASE(ItemSimRecommender):
    """The main class for EASE models.

    Main definition of attributes and data
    preparation shared between all implementations.

    Args:
        params (dict): The dictionary with the model params.
        *args (Any): Argument for PyTorch nn.Module.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.
    """

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
        self._name = "EASE"


@model_registry.register(name="EASE")
class EASE_Classic(EASE):
    """Implementation of EASE algorithm from
        Embarrassingly Shallow Autoencoders for Sparse Data 2019.

    For further details, check the `paper <https://arxiv.org/abs/1905.03375>`_.

    Attributes:
        l2 (float): The normalization value.
    """

    l2: float

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        The training will be conducted on the sparse representation of the interactions.
        This is the classic implementation for the EASE model.
        During the train a similarity matrix {item x item} will be learned.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        # The classic implementation follows the original paper
        X = interactions.get_sparse()

        G = X.T @ X + self.l2 * np.identity(X.shape[1])
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        self.item_similarity = nn.Parameter(torch.tensor(B, dtype=torch.float32))

        if report_fn is not None:
            report_fn(self)


@model_registry.register(name="EASE", implementation="Elliot")
class EASE_Elliot(EASE):
    """Implementation of EASE algorithm from
        Embarrassingly Shallow Autoencoders for Sparse Data 2019.

    This implementation was revised in the original Elliot framework.

    Attributes:
        l2 (float): The normalization value.
    """

    l2: float

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        The training will be conducted on the sparse representation of the interactions.
        This implementation was revised in the original Elliot framework.
        During the train a similarity matrix {item x item} will be learned.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        X = interactions.get_sparse()

        # The 'elliot' implementation add a popularity penalization
        B = safe_sparse_dot(X.T, X, dense_output=True)

        # Find diagonal indices and define item popularity to use as penalization
        diagonal_indices = np.diag_indices(B.shape[0])
        item_popularity = np.ediff1d(X.tocsc().indptr)

        # Penalize item on the diagonal with l2 norm and popularity
        B[diagonal_indices] = item_popularity + self.l2

        # Inverse and normalization
        P = np.linalg.inv(B)
        B = P / (-np.diag(P))

        # Remove diagonal items as in the classic implementation
        B[diagonal_indices] = 0.0

        self.item_similarity = nn.Parameter(torch.tensor(B, dtype=torch.float32))

        if report_fn is not None:
            report_fn(self)
