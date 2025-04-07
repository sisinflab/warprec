from typing import Any

import torch
import numpy as np
from torch import Tensor, nn
from scipy.sparse import csr_matrix
from elliotwo.data.dataset import Interactions
from elliotwo.recommenders.base_recommender import ItemSimilarityRecommender
from elliotwo.utils.registry import model_registry, similarities_registry


@model_registry.register(name="ItemKNN")
class ItemKNN(ItemSimilarityRecommender):
    """Implementation of ItemKNN algorithm from
        Amazon.com recommendations: item-to-item collaborative filtering 2003.

    For further details, check the `paper <http://ieeexplore.ieee.org/document/1167344/>`_.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
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

    def __init__(self, params: dict, info: dict, *args: Any, **kwargs: Any):
        super().__init__(params, info, *args, **kwargs)
        self._name = "ItemKNN"

    def fit(self, interactions: Interactions, *args: Any, **kwargs: Any):
        """During training we will compute the B similarity matrix {item x item}.

        Args:
            interactions (Interactions): The interactions that will be
                learned by the model.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        X = interactions.get_sparse()
        similarity = similarities_registry.get(self.similarity)

        # Apply normalization of interactions if requested
        if self.normalize:
            X = self._normalize(X)

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X.T)).to(self._device)

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(sim_matrix, self.k)

        # Update item_similarity with a new nn.Parameter
        self.item_similarity = nn.Parameter(filtered_sim_matrix.to(self._device))

    def _normalize(self, X: csr_matrix) -> csr_matrix:
        """Normalize matrix rows to unit length.

        Args:
            X (csr_matrix): The matrix to normalize.

        Returns:
            csr_matrix: The normalized matrix.
        """
        norms = np.sqrt(X.power(2).sum(axis=1))
        norms[norms == 0] = 1e-10
        return X.multiply(1 / norms)

    def _apply_topk_filtering(self, sim_matrix: Tensor, k: int) -> Tensor:
        """Keep only top-k similarities per item.

        Args:
            sim_matrix (Tensor): The similarity tensor to filter.
            k (int): The top k values to filter.

        Returns:
            Tensor: The filtered similarity tensor.
        """
        # Safety check for k size
        k = min(k, sim_matrix.size(1) - 1)

        # Get top-k values and indices
        values, indices = torch.topk(sim_matrix, k=k, dim=1)

        # Create sparse similarity matrix with top-k values
        return torch.zeros_like(sim_matrix).scatter_(1, indices, values)
