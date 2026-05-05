# pylint: disable = R0801, E1102
from typing import Any, Optional

import numpy as np
from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.utils.registry import model_registry


@model_registry.register(name="ADMMSlim")
class ADMMSlim(ItemSimRecommender):
    """Implementation of ADMMSlim algorithm from
        ADMM SLIM: Sparse Recommendations for Many Users 2020.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
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

    @classmethod
    def estimate_space(
        cls,
        params: dict,
        info: dict,
        interactions: Optional[Interactions] = None,
        **kwargs: Any,
    ) -> dict:
        interactions = cls._require_interactions_for_estimate(
            interactions, cls.__name__
        )
        X = interactions.get_sparse()
        n_users = info["n_users"]
        n_items = info["n_items"]

        train_matrix_mb = cls._sparse_size_mb(X)
        item_means_mb = cls._dense_size_mb((n_items,), np.float64)
        zero_mean_mb = (
            cls._dense_size_mb((n_users, n_items), np.float64)
            if params.get("center_columns")
            else 0.0
        )
        dense_item_float64_mb = cls._dense_size_mb((n_items, n_items), np.float64)
        dense_item_float32_mb = cls._dense_size_mb((n_items, n_items), np.float32)
        gamma_mb = cls._dense_size_mb((n_items,), np.float32)
        resident_mb = train_matrix_mb + item_means_mb

        build_gram_peak_mb = resident_mb + zero_mean_mb + dense_item_float64_mb
        diag_peak_mb = resident_mb + dense_item_float64_mb + 3 * dense_item_float64_mb
        inverse_peak_mb = (
            resident_mb + 3 * dense_item_float64_mb + dense_item_float32_mb
        )
        loop_peak_mb = resident_mb + 7 * dense_item_float32_mb + gamma_mb

        train_ram_mb = cls._peak_size_mb(
            build_gram_peak_mb,
            diag_peak_mb,
            inverse_peak_mb,
            loop_peak_mb,
        )
        return {
            "train_ram_mb": train_ram_mb,
            "notes": "ADMMSlim analytical train-space estimate",
        }

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, interactions, *args, seed=seed, **kwargs)

        X = self.train_matrix

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
        ) + self.rho * np.identity(self.n_items)
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
