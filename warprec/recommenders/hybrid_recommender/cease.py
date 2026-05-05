# pylint: disable = R0801, E1102
from typing import Any, Optional

import numpy as np
from scipy.sparse import vstack
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="CEASE")
class CEASE(ItemSimRecommender):
    """Implementation of CEASE algorithm from
        Closed-Form Models for Collaborative Filtering with Side-Information 2020.

    Args:
        params (dict): The dictionary with the model params.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        l2 (float): The normalization value.
        alpha (float): The alpha constant value.
    """

    l2: float
    alpha: float

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
        item_profile = interactions.get_side_sparse()
        if item_profile is None:
            raise ValueError("CEASE requires side information to estimate space.")

        n_items = info["n_items"]
        extended_nnz = X.nnz + item_profile.nnz
        extended_matrix_mb = cls._compressed_sparse_size_mb(
            nnz=extended_nnz,
            ptr_len=X.shape[0] + item_profile.shape[1] + 1,
            data_dtype=X.dtype,
        )
        gram_matrix_mb = cls._estimated_sparse_square_size_mb(
            source_nnz=extended_nnz,
            side_len=n_items,
            data_dtype=X.dtype,
        )
        dense_conversion_mb = cls._dense_size_mb((n_items, n_items), X.dtype)
        dense_inverse_mb = cls._dense_size_mb((n_items, n_items), np.float64)

        regularization_peak_mb = (
            extended_matrix_mb
            + gram_matrix_mb
            + dense_conversion_mb
            + 2 * dense_inverse_mb
        )
        inverse_peak_mb = dense_conversion_mb + 4 * dense_inverse_mb

        train_ram_mb = cls._peak_size_mb(
            regularization_peak_mb,
            inverse_peak_mb,
        )
        return {
            "train_ram_mb": train_ram_mb,
            "notes": "CEASE analytical train-space estimate",
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
        item_profile = interactions.get_side_sparse()

        # Extend interaction matrix
        X_extended = vstack((X, item_profile.T * self.alpha))

        G = X_extended.T @ X_extended + self.l2 * np.identity(X_extended.shape[1])
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        self.item_similarity = B
