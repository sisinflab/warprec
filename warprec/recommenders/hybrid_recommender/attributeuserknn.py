# pylint: disable = R0801, E1102, C0301
from typing import Any, Optional

import torch
import numpy as np
from torch import Tensor
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry, similarities_registry


@model_registry.register(name="AttributeUserKNN")
class AttributeUserKNN(Recommender):
    """Implementation of AttributeUserKNN algorithm from
        MyMediaLite: A free recommender system library 2011.

    For further details, please refer to the
        `paper <https://www.researchgate.net/publication/221141162_MyMediaLite_A_free_recommender_system_library>`_.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        k (int): Number of nearest neighbors.
        similarity (str): Similarity measure.
        user_profile (str): The computation of the user profile.
    """

    k: int
    similarity: str
    user_profile: str

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        X_inter = interactions.get_sparse()
        X_feat = interactions.get_side_sparse()
        similarity = similarities_registry.get(self.similarity)

        # Compute user profile
        X_profile = X_inter @ X_feat

        # Compute tfidf profile if requested
        if self.user_profile == "tfidf":
            X_profile = self._compute_user_tfidf(X_profile)

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X_profile))

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(sim_matrix, self.k)

        # Update item_similarity
        self.user_similarity = filtered_sim_matrix.numpy()

    def _compute_user_tfidf(self, user_profile: csr_matrix) -> csr_matrix:
        """Computes TF-IDF for user features.

        Args:
            user_profile (csr_matrix): The profile of the users.

        Returns:
            csr_matrix: The computed TF-IDF for users.
        """
        # Convert to average instead of sum
        user_counts = user_profile.sum(axis=1).A.ravel()
        user_counts[user_counts == 0] = 1  # Avoid division by zero
        user_profile = user_profile.multiply(1 / user_counts[:, np.newaxis])

        # L2 normalize
        return normalize(user_profile, norm="l2", axis=1)

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of B@X where B is a {user x user} similarity matrix.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.

        Raises:
            ValueError: If the 'train_sparse' keyword argument is not provided.
        """
        # Get train batch from kwargs
        train_sparse: Optional[csr_matrix] = kwargs.get("train_sparse")
        if train_sparse is None:
            raise ValueError(
                "predict() for AttributeUserKNN requires 'train_sparse' as a keyword argument."
            )

        # Compute predictions and convert to Tensor
        predictions = self.user_similarity[user_indices.cpu(), :] @ train_sparse
        predictions = torch.from_numpy(predictions)

        if item_indices is None:
            # Case 'full': prediction on all items
            return predictions  # [batch_size, n_items]

        # Case 'sampled': prediction on a sampled set of items
        return predictions.gather(
            1,
            item_indices.to(predictions.device).clamp(
                max=self.n_items - 1
            ),  # [batch_size, pad_seq]
        )
