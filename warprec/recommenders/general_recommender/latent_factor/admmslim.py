# pylint: disable = R0801, E1102
from typing import Optional, Callable, Any

import torch
import numpy as np
from torch import Tensor, nn
from scipy.sparse import csr_matrix
from warprec.data.dataset import Interactions
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.utils.registry import model_registry


@model_registry.register(name="ADMMSlim")
class ADMMSlim(ItemSimRecommender):
    """Implementation of ADMMSlim algorithm from
        ADMM SLIM: Sparse Recommendations for Many Users 2020.

    For further details, check the `paper <https://doi.org/10.1145/3336191.3371774>`_.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        lambda_1 (float): The first regularization parameter.
        lambda_2 (float): The second regularization parameter.
        alpha (float): The alpha parameter for the item means.
        rho (float): The rho parameter for the ADMM algorithm.
        it (int): The number of iterations for the ADMM algorithm.
        positive_only (bool): Wether or not to keep the similarity matrix positive.
        center_columns (bool): Wether or not to center the columns of the interactions.
    """

    lambda_1: float
    lambda_2: float
    alpha: float
    rho: float
    it: int
    positive_only: bool
    center_columns: bool

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
        self._name = "ADMMSlim"
        self.item_means = None

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        The training will be conducted on the sparse representation of the interactions.
        During the train a similarity matrix {item x item} will be learned.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        X = interactions.get_sparse()

        # Calculate the item means
        self.item_means = X.mean(axis=0).getA1()

        if self.center_columns:
            # Center the columns of the interactions
            # This is memory expensive, on large dataset it's better to
            # leave this parameter to false
            zero_mean_X = X.toarray() - self.item_means
            G = zero_mean_X.T @ zero_mean_X

            del zero_mean_X  # We remove zero_mean_X cause of high cost in memory
        else:
            G = (X.T @ X).toarray()

        # Pre-compute values for ADMMSlim algorithm
        diag = self.lambda_2 * np.diag(
            np.power(self.item_means, self.alpha)
        ) + self.rho * np.identity(self.items)
        P = np.linalg.inv(G + diag).astype(np.float32)
        B_aux = (P @ G).astype(np.float32)

        # Initialize
        Gamma = np.zeros_like(G, dtype=np.float32)
        C = np.zeros_like(G, dtype=np.float32)

        del diag, G  # We also remove G cause of high cost in memory

        # ADMM iterations
        for _ in range(self.it):
            B_tilde = B_aux + P @ (self.rho * C - Gamma)
            gamma = np.diag(B_tilde) / (np.diag(P) + 1e-7)
            B = B_tilde - P * gamma
            C = self._soft_threshold(B + Gamma / self.rho, self.lambda_1 / self.rho)
            if self.positive_only:
                C = np.maximum(C, 0)
            Gamma += self.rho * (B - C)

        # Update item_similarity with a new nn.Parameter
        self.item_similarity = nn.Parameter(torch.from_numpy(C))

        if report_fn is not None:
            report_fn(self)

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Additionally, if the center_columns parameter has been set to True,
        the item means will be added to the result.

        Args:
            interaction_matrix (csr_matrix): The matrix containing the
                pairs of interactions to evaluate.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """

        if self.center_columns:
            # If centering was applied, then we center the interactions also
            # then add back the means
            r = (
                interaction_matrix - self.item_means
            ) @ self.item_similarity.detach().numpy() + self.item_means
        else:
            r = interaction_matrix @ self.item_similarity.detach().numpy()

        # Masking interaction already seen in train
        r[interaction_matrix.nonzero()] = -torch.inf
        return torch.from_numpy(r).to(self._device)

    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        return (np.abs(x) > threshold) * (np.abs(x) - threshold) * np.sign(x)
