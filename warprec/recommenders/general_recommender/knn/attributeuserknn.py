# pylint: disable = R0801, E1102
from typing import Any

import torch
import numpy as np
from torch import Tensor
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from warprec.data.dataset import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry, similarities_registry


@model_registry.register(name="AttributeUserKNN")
class AttributeUserKNN(Recommender):
    """Implementation of AttributeUserKNN algorithm from
        MyMediaLite: A free recommender system library 2011.

    For further details, check the `paper <https://www.researchgate.net/publication/221141162_MyMediaLite_A_free_recommender_system_library>`_.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
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
    def predict_full(
        self,
        user_indices: Tensor,
        train_sparse: csr_matrix,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of B@X where B is a {user x user} similarity matrix.

        Args:
            user_indices (Tensor): The batch of user indices.
            train_sparse (csr_matrix): The full of train sparse
                interaction matrix.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Compute predictions and convert to Tensor
        predictions = self.user_similarity[user_indices.cpu(), :] @ train_sparse
        predictions = torch.from_numpy(predictions)
        return predictions.to(self._device)

    @torch.no_grad()
    def predict_sampled(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
        train_sparse: csr_matrix,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of B@X where B is a {user x user} similarity matrix.

        This method will produce predictions only for given item indices.

        Args:
            user_indices (Tensor): The batch of user indices.
            item_indices (Tensor): The batch of item indices to sample.
            train_sparse (csr_matrix): The full train sparse
                interaction matrix.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x pad_seq}.
        """
        # Compute predictions
        predictions = self.user_similarity[user_indices.cpu(), :] @ train_sparse

        # Convert to Tensor and gather only required indices
        predictions = torch.from_numpy(predictions).to(self._device)
        predictions = predictions.gather(
            1, item_indices.clamp(min=0)
        )  # [batch_size, pad_seq]
        return predictions
