# pylint: disable = R0801, E1102
from typing import Any, Optional

import numpy as np

from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.utils.registry import model_registry


@model_registry.register(name="AddEASE")
class AddEASE(ItemSimRecommender):
    """Implementation of AddEASE algorithm from
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
            raise ValueError("AddEASE requires side information to estimate space.")

        n_items = info["n_items"]
        collab_sparse_gram_mb = cls._estimated_sparse_square_size_mb(
            source_nnz=X.nnz,
            side_len=n_items,
            data_dtype=X.dtype,
        )
        side_sparse_gram_mb = cls._estimated_sparse_square_size_mb(
            source_nnz=item_profile.nnz,
            side_len=n_items,
            data_dtype=item_profile.dtype,
        )
        collab_dense_conversion_mb = cls._dense_size_mb((n_items, n_items), X.dtype)
        side_dense_conversion_mb = cls._dense_size_mb(
            (n_items, n_items), item_profile.dtype
        )
        dense_inverse_mb = cls._dense_size_mb((n_items, n_items), np.float64)

        collab_peak_mb = cls._peak_size_mb(
            collab_sparse_gram_mb + collab_dense_conversion_mb + 2 * dense_inverse_mb,
            collab_dense_conversion_mb + 4 * dense_inverse_mb,
        )
        side_peak_mb = cls._peak_size_mb(
            side_sparse_gram_mb + side_dense_conversion_mb + 2 * dense_inverse_mb,
            side_dense_conversion_mb + 4 * dense_inverse_mb,
        )
        merge_peak_mb = 3 * dense_inverse_mb

        train_ram_mb = cls._peak_size_mb(
            collab_peak_mb,
            side_peak_mb,
            merge_peak_mb,
        )
        return {
            "train_ram_mb": train_ram_mb,
            "notes": "AddEASE analytical train-space estimate",
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

        # Fist solution
        G = X.T @ X + self.l2 * np.identity(X.shape[1])
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        # Second solution
        item_profile = interactions.get_side_sparse()
        P = item_profile @ item_profile.T + self.l2 * np.identity(item_profile.shape[0])
        U = np.linalg.inv(P)
        U /= -np.diag(U)
        np.fill_diagonal(U, 0.0)

        # Linear combination
        self.item_similarity = self.alpha * B + (1 - self.alpha) * U
