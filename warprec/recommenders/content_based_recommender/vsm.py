# pylint: disable = R0801, E1102, R0401
from typing import Any, Optional

import torch
import numpy as np
from torch import Tensor
from scipy.sparse import csr_matrix, diags
from sklearn.preprocessing import normalize

from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry, similarities_registry


@model_registry.register(name="VSM")
class VSM(Recommender):
    """Implementation of VSM algorithm from
        Linked Open Data to support Content-based Recommender Systems 2012.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        similarity (str): Similarity measure.
        user_profile (str): The computation of the user profile.
        item_profile (str): The computation of the item profile.
    """

    similarity: str
    user_profile: str
    item_profile: str

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
            raise ValueError("VSM requires side information to estimate space.")

        n_users = info["n_users"]
        n_items = info["n_items"]
        n_features = item_profile.shape[1]

        avg_features_per_item = item_profile.nnz / max(n_items, 1)
        user_profile_nnz = int(
            min(n_users * n_features, np.ceil(X.nnz * avg_features_per_item))
        )

        train_matrix_mb = cls._sparse_size_mb(X)
        item_profile_mb = cls._sparse_size_mb(item_profile)
        user_profile_mb = cls._compressed_sparse_size_mb(
            nnz=user_profile_nnz,
            ptr_len=n_users + 1,
            data_dtype=X.dtype,
        )
        resident_item_profile_mb = item_profile_mb
        train_ram_mb = cls._peak_size_mb(
            train_matrix_mb + resident_item_profile_mb,
            train_matrix_mb + resident_item_profile_mb + user_profile_mb,
        )

        if params.get("item_profile") == "tfidf":
            item_idf_mb = cls._dense_size_mb((n_features,), np.float64)
            idf_diag_mb = cls._compressed_sparse_size_mb(
                nnz=n_features,
                ptr_len=n_features + 1,
                data_dtype=np.float64,
            )
            train_ram_mb = cls._peak_size_mb(
                train_ram_mb,
                train_matrix_mb + 2 * item_profile_mb + item_idf_mb + idf_diag_mb,
                train_matrix_mb + item_profile_mb + user_profile_mb,
            )
            resident_item_profile_mb = item_profile_mb

        if params.get("user_profile") == "tfidf":
            train_ram_mb = cls._peak_size_mb(
                train_ram_mb,
                train_matrix_mb
                + resident_item_profile_mb
                + 2 * user_profile_mb
                + cls._dense_size_mb((n_users,), np.float64),
            )

        return {
            "train_ram_mb": train_ram_mb,
            "notes": "VSM analytical train-space estimate",
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

        # Get data from interactions
        X = interactions.get_sparse()  # [user x item]
        item_profile = interactions.get_side_sparse()  # [item x side]
        self.sim_function = similarities_registry.get(self.similarity)

        if self.item_profile == "tfidf":
            # Compute TF-IDF
            item_profile = self._compute_item_tfidf(item_profile)

        # Compute binary user profile
        user_profile = X @ item_profile  # [user x side]

        if self.user_profile == "tfidf":
            user_profile = self._compute_user_tfidf(user_profile)

        # Save profiles
        self.i_profile = item_profile
        self.u_profile = user_profile

    def _compute_item_tfidf(self, item_profile: csr_matrix) -> csr_matrix:
        """Computes TF-IDF for item features.

        Args:
            item_profile (csr_matrix): The profile of the items.

        Returns:
            csr_matrix: The computed TF-IDF for items.
        """
        n_items = item_profile.shape[0]

        # Document Frequency (per feature)
        df = np.diff(item_profile.tocsc().indptr)

        # IDF with smoothing
        idf = np.log((n_items + 1) / (df + 1)) + 1

        # TF remains as raw counts
        tf = item_profile.copy()

        # TF-IDF calculation
        idf_diag = diags(idf)
        tfidf = tf @ idf_diag

        # L2 normalize
        return normalize(tfidf, norm="l2", axis=1)

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
        """Prediction using the learned embeddings.

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
        predictions_numpy = self.sim_function.compute(
            self.u_profile[user_indices.tolist()], self.i_profile
        )
        predictions = torch.from_numpy(predictions_numpy)

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
