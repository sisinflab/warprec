from typing import Any

import torch
import numpy as np
from torch import nn
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
    """

    k: int
    similarity: str
    # normalize: bool

    def __init__(self, params: dict, items: int, *args: Any, **kwargs: Any):
        super().__init__(params, items, *args, **kwargs)
        self._name = "ItemKNN"

    def fit(self, interactions: Interactions, *args: Any, **kwargs: Any) -> None:
        """Compute item similarity matrix using specified similarity measure"""
        X = self._preprocess_matrix(interactions.get_sparse())
        similarity = similarities_registry.get(self.similarity)
        self.item_similarity = nn.Parameter(
            torch.from_numpy(similarity.compute(X.T))
        ).to(self._device)
        self._apply_topk_filtering()

    def _preprocess_matrix(self, X: csr_matrix) -> csr_matrix:
        """Apply normalization"""
        # if self.normalize:
        #    X = self._normalize(X)
        return X

    def _normalize(self, X: csr_matrix) -> csr_matrix:
        """Normalize matrix rows to unit length"""
        norms = np.sqrt(X.power(2).sum(axis=1))
        norms[norms == 0] = 1e-10
        return X.multiply(1 / norms)

    def _apply_topk_filtering(self):
        """Keep only top-k similarities per item"""
        sim_matrix = self.item_similarity.detach().numpy()
        for i in range(sim_matrix.shape[0]):
            row = sim_matrix[i]
            top_k_indices = np.argpartition(row, -self.k)[-self.k :]
            mask = np.zeros_like(row)
            mask[top_k_indices] = 1
            sim_matrix[i] *= mask
        self.item_similarity = nn.Parameter(torch.from_numpy(sim_matrix)).to(
            self._device
        )
