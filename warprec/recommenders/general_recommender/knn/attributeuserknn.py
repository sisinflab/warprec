# pylint: disable = R0801, E1102
from typing import Optional, Callable, Any

import torch
import numpy as np
from torch import Tensor, nn
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
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the users value was not passed through the info dict.

    Attributes:
        k (int): Number of nearest neighbors.
        similarity (str): Similarity measure.
        user_profile (str): The computation of the user profile.
        normalize (bool): Wether or not to normalize the interactions.
    """

    k: int
    similarity: str
    user_profile: str
    normalize: bool

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
        self._name = "AttributeUserKNN"
        users = info.get("users", None)
        if not users:
            raise ValueError(
                "Users value must be provided to correctly initialize the model."
            )
        # Model initialization
        self.user_similarity = nn.Parameter(torch.rand(users, users))

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        The training will be conducted on the sparse representation of the features.
        During the train a similarity matrix {user x user} will be learned.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        X_inter = interactions.get_sparse()
        X_feat = interactions.get_side_sparse()
        similarity = similarities_registry.get(self.similarity)

        # Compute user profile
        X_profile = X_inter @ X_feat

        # Apply normalization of profiles if requested
        if self.normalize:
            X_profile = self._normalize(X_profile)

        # Compute tfidf profile if requested
        if self.user_profile == "tfidf":
            X_profile = self._compute_user_tfidf(X_profile)

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X_profile))

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(sim_matrix, self.k)

        # Update item_similarity with a new nn.Parameter
        self.user_similarity = nn.Parameter(filtered_sim_matrix)

        if report_fn is not None:
            report_fn(self)

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction in the form of B@X where B is a {user x user} similarity matrix.

        Args:
            interaction_matrix (csr_matrix): The interactions matrix
                that will be used to predict.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        start_idx = kwargs.get("start", 0)
        end_idx = kwargs.get("end", interaction_matrix.shape[0])
        r = (
            self.user_similarity.detach().numpy()[start_idx:end_idx, start_idx:end_idx]
            @ interaction_matrix
        )

        # Masking interaction already seen in train
        r[interaction_matrix.nonzero()] = -torch.inf
        return torch.from_numpy(r).to(self._device)

    def forward(self, *args, **kwargs):
        """Forward method is empty because we don't need
        back propagation.
        """

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
