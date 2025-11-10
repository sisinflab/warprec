# pylint: disable = R0801, E1102
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from warprec.data.entities import Interactions
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry, similarities_registry


@model_registry.register(name="UserKNN")
class UserKNN(Recommender):
    """Implementation of UserKNN algorithm from
        GroupLens: an open architecture for collaborative filtering of netnews 1994.

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/192844.192905>`_.

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
    """

    k: int
    similarity: str

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
            params, interactions, device=device, seed=seed, *args, **kwargs
        )

        X = interactions.get_sparse()
        similarity = similarities_registry.get(self.similarity)

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X))

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(sim_matrix, self.k)

        # Update item_similarity
        self.user_similarity = filtered_sim_matrix.numpy()

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
