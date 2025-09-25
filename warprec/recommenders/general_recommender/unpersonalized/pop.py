# pylint: disable = R0801, E1102
from typing import Any

import torch
from torch import Tensor
from warprec.recommenders.base_recommender import Recommender
from warprec.data.dataset import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="Pop")
class Pop(Recommender):
    """Definition of Popularity unpersonalized model.

    This model will recommend items based on their popularity,
    ensuring that previously seen items are not recommended again.

    Args:
        params (dict): The dictionary with the model params.
        interactions (Interactions): The training interactions.
        *args (Any): Argument for PyTorch nn.Module.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.
    """

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
        self._name = "Pop"

        X = interactions.get_sparse()

        # Count the number of items to define the popularity
        self.popularity = torch.tensor(
            X.sum(axis=0).A1, device=self._device, dtype=torch.float32
        )
        # Count the total number of interactions
        self.item_count = torch.tensor(
            X.sum(), device=self._device, dtype=torch.float32
        )

    @torch.no_grad()
    def predict_full(
        self,
        user_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using a normalized popularity value.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        batch_size = user_indices.size(0)

        # Normalize popularity by the total number of interactions
        # Add epsilon to avoid division by zero if there are no interactions
        normalized_popularity = self.popularity / (self.item_count + 1e-6)

        # Repeat the popularity scores for each user in the batch
        predictions = normalized_popularity.repeat(batch_size, 1).to(self._device)
        return predictions

    @torch.no_grad()
    def predict_sampled(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using a random number generator.

        Args:
            user_indices (Tensor): The batch of user indices.
            item_indices (Tensor): The batch of item indices.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x pad_seq}.
        """
        # Normalize popularity by the total number of interactions
        # Add epsilon to avoid division by zero if there are no interactions
        normalized_popularity = self.popularity / (self.item_count + 1e-6)

        # Retrieve the popularity scores for the sampled items
        # Clamp item_indices to avoid out-of-bounds indexing with -1
        predictions = normalized_popularity[item_indices.clamp(min=0)]
        return predictions.to(self._device)
