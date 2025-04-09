# pylint: disable = R0801, E1102
from typing import Any

import torch
import scipy.sparse as sp
from torch import nn
from torch import Tensor
from scipy.sparse import csr_matrix
from sklearn.linear_model import ElasticNet
from elliotwo.recommenders.base_recommender import Recommender
from elliotwo.data.dataset import Interactions
from elliotwo.utils.registry import model_registry


@model_registry.register(name="Slim")
class Slim(Recommender):
    """Implementation of Slim model from
        Sparse Linear Methods for Top-N Recommender Systems 2011.

    For further details, check the `paper <https://ieeexplore.ieee.org/document/6137254>`_.

    Args:
        params (dict): The dictionary with the model params.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items value was not passed through the info dict.

    Attributes:
        l1 (float): The normalization value.
        alpha (float): The alpha multiplication constant value.
    """

    l1: float
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
        super().__init__(params, device=device, seed=seed, *args, **kwargs)
        self._name = "Slim"
        items = info.get("items", None)
        if not items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        # Model initialization
        self.item_similarity = nn.Parameter(torch.rand(items, items)).to(self._device)

    def fit(self, interactions: Interactions, *args: Any, **kwargs: Any):
        """Main train method.

        The training will be conducted for each user using ElasticNet.
        This train loop might not scale much for large dataset with a lot of users.
        During the train a similarity matrix {item x item} will be learned.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.
        """
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

    @torch.no_grad()
    def predict(
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

    def forward(self, *args, **kwargs):
        """Forward method is empty because we don't need
        back propagation.
        """
        pass
