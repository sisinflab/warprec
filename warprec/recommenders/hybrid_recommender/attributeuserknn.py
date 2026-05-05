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
        X_inter = interactions.get_sparse()
        X_feat = interactions.get_side_sparse()
        if X_feat is None:
            raise ValueError(
                "AttributeUserKNN requires side information to estimate space."
            )

        n_users = info["n_users"]
        n_items = info["n_items"]
        n_features = X_feat.shape[1]

        avg_features_per_item = X_feat.nnz / max(n_items, 1)
        profile_nnz = int(
            min(n_users * n_features, np.ceil(X_inter.nnz * avg_features_per_item))
        )

        train_matrix_mb = cls._sparse_size_mb(X_inter)
        feature_matrix_mb = cls._sparse_size_mb(X_feat)
        profile_matrix_mb = cls._compressed_sparse_size_mb(
            nnz=profile_nnz,
            ptr_len=n_users + 1,
            data_dtype=X_inter.dtype,
        )
        similarity_matrix_mb = cls._dense_size_mb((n_users, n_users), X_inter.dtype)

        train_ram_mb = cls._peak_size_mb(
            train_matrix_mb + feature_matrix_mb + profile_matrix_mb,
            train_matrix_mb + profile_matrix_mb + similarity_matrix_mb,
        )
        if params.get("user_profile") == "tfidf":
            train_ram_mb = cls._peak_size_mb(
                train_ram_mb,
                train_matrix_mb
                + feature_matrix_mb
                + 2 * profile_matrix_mb
                + cls._dense_size_mb((n_users,), np.float64),
            )

        return {
            "train_ram_mb": train_ram_mb,
            "notes": "AttributeUserKNN analytical train-space estimate",
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
        super().__init__(params, info, *args, seed=seed, **kwargs)

        # Store the training matrix for prediction
        self.train_matrix = interactions.get_sparse()

        X_inter = self.train_matrix
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
        """
        # Compute predictions and convert to Tensor
        predictions = self.user_similarity[user_indices.cpu(), :] @ self.train_matrix
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
