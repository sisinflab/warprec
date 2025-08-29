# pylint: disable = R0801, E1102
from typing import Any

import torch
import numpy as np
from torch import Tensor, nn
from scipy.sparse import csr_matrix, coo_matrix, diags
from sklearn.preprocessing import normalize
from warprec.data.dataset import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry, similarities_registry


@model_registry.register(name="VSM")
class VSM(Recommender):
    """Implementation of VSM algorithm from
        Linked Open Data to support Content-based Recommender Systems 2012.

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/2362499.2362501>`_.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        similarity (str): Similarity measure.
        user_profile (str): The computation of the user profile.
        item_profile (str): The computation of the item profile.
    """

    similarity: str
    user_profile: str
    item_profile: str

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
        self._name = "VSM"

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
        self.i_profile = nn.Parameter(self._scipy_sparse_to_torch_sparse(item_profile))
        self.u_profile = nn.Parameter(self._scipy_sparse_to_torch_sparse(user_profile))

    def _compute_item_tfidf(self, item_profile: csr_matrix) -> csr_matrix:
        """Computes TF-IDF for item features.

        Args:
            item_profile (csr_matrix): The profile of the items.

        Returns:
            csr_matrix: The computed TF-IDF for items.
        """
        n_items, _ = item_profile.shape

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

    def _scipy_sparse_to_torch_sparse(self, sparse_matrix: csr_matrix) -> Tensor:
        """Convert the sparse scipy to torch sparse.

        Args:
            sparse_matrix (csr_matrix): The sparse matrix in csr format.

        Returns:
            Tensor: The converted tensor.
        """
        coo_matrix = sparse_matrix.tocoo()
        indices = torch.tensor(np.vstack((coo_matrix.row, coo_matrix.col)))
        data = torch.tensor(coo_matrix.data)
        shape = coo_matrix.shape
        return torch.sparse_coo_tensor(indices, data, shape)

    def _torch_sparse_to_scipy_sparse(self, sparse_tensor: Tensor) -> csr_matrix:
        """Convert the sparse tensor to scipy sparse.

        Args:
            sparse_tensor (Tensor): The sparse tensor in csr format.

        Returns:
            csr_matrix: The converted matrix.
        """
        sparse_tensor = sparse_tensor.coalesce()
        indices = sparse_tensor.indices().numpy()
        values = sparse_tensor.values().detach().numpy()
        shape = sparse_tensor.shape
        scipy_coo = coo_matrix((values, (indices[0], indices[1])), shape=shape)
        return scipy_coo.tocsr()

    @torch.no_grad()
    def predict_full(
        self,
        train_batch: Tensor,
        user_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Args:
            train_batch (Tensor): The train batch of user interactions.
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Extract profiles and convert them to scipy
        user_profile = self._torch_sparse_to_scipy_sparse(self.u_profile)
        item_profile = self._torch_sparse_to_scipy_sparse(self.i_profile)

        # Compute similarity
        predictions_numpy = self.sim_function.compute(
            user_profile[user_indices.cpu().numpy()], item_profile
        )
        predictions = torch.from_numpy(predictions_numpy)

        # Masking interaction already seen in train
        predictions[train_batch != 0] = -torch.inf
        return predictions.to(self._device)

    @torch.no_grad()
    def predict_sampled(
        self,
        train_batch: Tensor,
        user_indices: Tensor,
        item_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        This method will produce predictions only for given item indices.

        Args:
            train_batch (Tensor): The train batch of user interactions.
            user_indices (Tensor): The batch of user indices.
            item_indices (Tensor): The batch of item indices to sample.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x pad_seq}.
        """
        # Extract profiles and convert them to scipy
        user_profile = self._torch_sparse_to_scipy_sparse(self.u_profile)
        item_profile = self._torch_sparse_to_scipy_sparse(self.i_profile)

        # Compute predictions and gather only sampled items
        predictions_numpy = self.sim_function.compute(
            user_profile[user_indices.cpu().numpy()], item_profile
        )
        predictions = torch.from_numpy(predictions_numpy)
        predictions = predictions.gather(
            1, item_indices.clamp(min=0)
        )  # [batch_size, pad_seq]

        # Mask padded indices
        predictions[item_indices == -1] = -torch.inf
        return predictions.to(self._device)
