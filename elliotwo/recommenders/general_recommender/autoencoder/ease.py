# pylint: disable = R0801, E1102
from typing import Any

import numpy as np
import torch
from torch import nn
from torch import Tensor
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import safe_sparse_dot
from elliotwo.recommenders.base_recommender import Recommender
from elliotwo.data.dataset import Interactions
from elliotwo.utils.registry import model_registry


class EASE(Recommender):
    """The main class for EASE models.

    Main definition of attributes and data
    preparation shared between all implementations.

    Args:
        params (dict): The dictionary with the model params.
        *args (Any): Argument for PyTorch nn.Module.
        device (str): The device used for tensor operations.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Raises:
        ValueError: If the items value was not passed through the info dict.
    """

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, *args, **kwargs)
        self._name = "EASE"
        items = info.get("items", None)
        if not items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        # Model initialization
        self.item_similarity = nn.Parameter(torch.rand(items, items)).to(self._device)

    def forward(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Args:
            interaction_matrix (csr_matrix): The interactions matrix
                that will be used to predict.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        r = interaction_matrix @ self.item_similarity.detach().numpy()

        # Masking interaction already seen in train
        r[interaction_matrix.nonzero()] = -torch.inf
        return torch.from_numpy(r).to(self._device)


@model_registry.register(name="EASE")
class EASE_Classic(EASE):
    """Implementation of EASE algorithm from
        Embarrassingly Shallow Autoencoders for Sparse Data 2019.

    For further details, check the `paper <https://arxiv.org/abs/1905.03375>`_.

    Attributes:
        l2 (float): The normalization value.
    """

    l2: float

    def fit(self, interactions: Interactions, *args: Any, **kwargs: Any):
        """During training we will compute the B similarity matrix {item x item}.

        Args:
            interactions (Interactions): The interactions that will be
                learned by the model.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        # The classic implementation follows the original paper
        X = interactions.get_sparse()

        G = X.T @ X + self.l2 * np.identity(X.shape[1])
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        self.item_similarity = nn.Parameter(torch.tensor(B, dtype=torch.float32)).to(
            self._device
        )


@model_registry.register(name="EASE", implementation="Elliot")
class EASE_Elliot(EASE):
    """Implementation of EASE algorithm from
        Embarrassingly Shallow Autoencoders for Sparse Data 2019.

    This implementation was revised in the original Elliot framework.

    Attributes:
        l2 (float): The normalization value.
    """

    l2: float

    def fit(self, interactions: Interactions, *args: Any, **kwargs: Any):
        """During training we will compute the B similarity matrix {item x item}.

        Args:
            interactions (Interactions): The interactions that will be
                learned by the model.
            *args (Any): List of arguments.
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

        self.item_similarity = nn.Parameter(torch.tensor(B, dtype=torch.float32)).to(
            self._device
        )
