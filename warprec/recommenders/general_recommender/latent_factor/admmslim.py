# pylint: disable = R0801, E1102
from typing import Any

import numpy as np
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
        interactions (Interactions): The training interactions.
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
        self._name = "ADMMSlim"

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

        # Update item_similarity
        self.item_similarity = C

    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        return (np.abs(x) > threshold) * (np.abs(x) - threshold) * np.sign(x)
