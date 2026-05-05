# pylint: disable = R0801, E1102
from typing import Any, Optional

import numpy as np

from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.utils.registry import model_registry


@model_registry.register(name="EASE")
class EASE(ItemSimRecommender):
    """Implementation of EASE algorithm from
        Embarrassingly Shallow Autoencoders for Sparse Data 2019.

    Args:
        params (dict): The dictionary with the model params.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        l2 (float): The normalization value.
    """

    l2: float

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
        n_items = info["n_items"]

        gram_matrix_mb = cls._estimated_sparse_square_size_mb(
            source_nnz=X.nnz,
            side_len=n_items,
            data_dtype=X.dtype,
        )
        dense_conversion_mb = cls._dense_size_mb((n_items, n_items), X.dtype)
        dense_inverse_mb = cls._dense_size_mb((n_items, n_items), np.float64)

        regularization_peak_mb = (
            gram_matrix_mb + dense_conversion_mb + 2 * dense_inverse_mb
        )
        inverse_peak_mb = dense_conversion_mb + 4 * dense_inverse_mb

        return {
            "train_ram_mb": cls._peak_size_mb(
                regularization_peak_mb,
                inverse_peak_mb,
            ),
            "notes": "EASE analytical train-space estimate",
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
        G = X.T @ X + self.l2 * np.identity(X.shape[1])
        B = np.linalg.inv(G)
        B /= -np.diag(B)
        np.fill_diagonal(B, 0.0)

        self.item_similarity = B
