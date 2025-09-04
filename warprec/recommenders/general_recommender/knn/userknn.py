# pylint: disable = R0801, E1102
from typing import Any

import torch
from torch import Tensor, nn
from warprec.data.dataset import Interactions
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
        self._name = "UserKNN"

        X = interactions.get_sparse()
        similarity = similarities_registry.get(self.similarity)

        # Compute similarity matrix
        sim_matrix = torch.from_numpy(similarity.compute(X))

        # Compute top_k filtering
        filtered_sim_matrix = self._apply_topk_filtering(sim_matrix, self.k)

        # Update item_similarity with a new nn.Parameter
        self.user_similarity = nn.Parameter(filtered_sim_matrix)

    @torch.no_grad()
    def predict_full(
        self,
        train_batch: Tensor,
        user_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of B@X where B is a {user x user} similarity matrix.

        Args:
            train_batch (Tensor): The train batch of user interactions.
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        predictions = (
            self.user_similarity[user_indices, :][:, user_indices] @ train_batch
        )

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
        """Prediction in the form of B@X where B is a {user x user} similarity matrix.

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
        # Compute predictions and gather only sampled items
        predictions = (
            self.user_similarity[user_indices, :][:, user_indices] @ train_batch
        )
        predictions = predictions.gather(
            1, item_indices.clamp(min=0)
        )  # [batch_size, pad_seq]

        # Mask padded indices
        predictions[item_indices == -1] = -torch.inf
        return predictions.to(self._device)
