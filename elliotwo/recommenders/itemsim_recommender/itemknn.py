from typing import Any

import torch
import numpy as np
from torch import Tensor, nn
from scipy.sparse import csr_matrix
from elliotwo.data.dataset import Interactions
from elliotwo.recommenders.abstract_recommender import ItemSimilarityRecommender
from elliotwo.utils.registry import model_registry, similarities_registry


@model_registry.register(name="ItemKNN")
class ItemKNN(ItemSimilarityRecommender):
    """ItemKNN implementation supporting multiple similarity measures.

    Args:
        params (dict): Model parameters.
        items (int): Number of items in the dataset.
        *args (Any): Variable length argument list.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        k (int): Number of nearest neighbors.
        similarity (str): Similarity measure.
        normalize (bool): Wether or not to normalize the interactions.
    """

    k: int
    similarity: str
    normalize: bool

    def __init__(self, params: dict, items: int, *args: Any, **kwargs: Any):
        super().__init__(params, items, *args, **kwargs)
        self._name = "ItemKNN"

    def fit(self, interactions: Interactions, *args: Any, **kwargs: Any) -> None:
        """Compute item similarity matrix using specified similarity measure"""
        X = interactions.get_sparse()
        similarity = similarities_registry.get(self.similarity)

        # Apply normalization of interactions if requested
        if self.normalize:
            X = self._normalize(X)

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X.T)).to(self._device)

        # Compute top_k filtering
        self._apply_topk_filtering(sim_matrix)

    def _normalize(self, X: csr_matrix) -> csr_matrix:
        """Normalize matrix rows to unit length"""
        norms = np.sqrt(X.power(2).sum(axis=1))
        norms[norms == 0] = 1e-10
        return X.multiply(1 / norms)

    def _apply_topk_filtering(self, sim_matrix: Tensor):
        """Keep only top-k similarities per item"""
        # Safety check for k size
        k = min(self.k, sim_matrix.size(1) - 1)

        # Get top-k values and indices
        values, indices = torch.topk(sim_matrix, k=k, dim=1)

        # Create sparse similarity matrix with top-k values
        filtered_sim_matrix = torch.zeros_like(sim_matrix).scatter_(1, indices, values)

        # Update item_similarity with a new nn.Parameter
        self.item_similarity = nn.Parameter(filtered_sim_matrix.to(self._device))
