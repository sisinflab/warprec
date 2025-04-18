# pylint: disable = R0801, E1102
from typing import Optional, Callable, Any

import torch
import numpy as np
from torch import Tensor, nn
from scipy.sparse import csr_matrix, coo_matrix, diags
from sklearn.preprocessing import normalize
from elliotwo.data.dataset import Interactions
from elliotwo.recommenders.base_recommender import Recommender
from elliotwo.utils.registry import model_registry, similarities_registry


@model_registry.register(name="VSM")
class VSM(Recommender):
    """Implementation of VSM algorithm from
        Linked Open Data to support Content-based Recommender Systems 2012.

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/2362499.2362501>`_.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        similarity (str): Similarity measure.
        user_profile (str): The computation of the user profile.
        item_profile (str): The computation of the item profile.

    Raises:
        ValueError: If the items, users or features value was not passed through the info dict.
    """

    similarity: str
    user_profile: str
    item_profile: str

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, info=info, *args, **kwargs)
        self._name = "VSM"

        users = info.get("users", None)
        if not users:
            raise ValueError(
                "Users value must be provided to correctly initialize the model."
            )
        items = info.get("items", None)
        if not items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        features = info.get("features", None)
        if not features:
            raise ValueError(
                "Features value must be provided to correctly initialize the model."
            )

        # Initialize similarity
        self.sim_function = similarities_registry.get(self.similarity)

        # Model initialization
        self.i_profile = nn.Parameter(torch.sparse_coo_tensor(size=(items, features)))  # type: ignore[call-arg]
        self.u_profile = nn.Parameter(torch.sparse_coo_tensor(size=(users, features)))  # type: ignore[call-arg]

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        # Get data from interactions
        X = interactions.get_sparse()  # [user x item]
        item_profile = interactions.get_side_sparse()  # [item x side]

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

        if report_fn is not None:
            report_fn(self)

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Args:
            interaction_matrix (csr_matrix): The matrix containing the
                pairs of interactions to evaluate.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        start_idx = kwargs.get("start", 0)
        end_idx = kwargs.get("end", interaction_matrix.shape[0])

        # Extract profiles and convert them to scipy
        user_profile = self._torch_sparse_to_scipy_sparse(self.u_profile)
        item_profile = self._torch_sparse_to_scipy_sparse(self.i_profile)

        # Compute similarity
        r = self.sim_function.compute(user_profile[start_idx:end_idx], item_profile)

        # Masking interaction already seen in train
        r[interaction_matrix.nonzero()] = -torch.inf
        return torch.from_numpy(r).to(self._device)

    def forward(self, *args, **kwargs):
        """Forward method is empty because we don't need
        back propagation.
        """

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
