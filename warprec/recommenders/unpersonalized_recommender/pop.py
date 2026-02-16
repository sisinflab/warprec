# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
from torch import Tensor
from warprec.recommenders.base_recommender import Recommender
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="Pop")
class Pop(Recommender):
    """Definition of Popularity unpersonalized model.

    This model will recommend items based on their popularity,
    ensuring that previously seen items are not recommended again.

    Args:
        params (dict): The dictionary with the model params.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        normalized_popularity (Tensor): The lookup tensor for normalized
            popularity.
    """

    normalized_popularity: Tensor

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

        X = interactions.get_sparse()

        # Count the number of items to define the popularity
        popularity = torch.tensor(X.sum(axis=0).A1, dtype=torch.float32)
        # Count the total number of interactions
        item_count = torch.tensor(X.sum(), dtype=torch.float32)

        # Normalize popularity by the total number of interactions
        # Add epsilon to avoid division by zero if there are no interactions
        norm_pop = popularity / (item_count + 1e-6)
        self.register_buffer("normalized_popularity", norm_pop)

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using a normalized popularity value.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        if item_indices is None:
            # Case 'full': prediction on all items
            batch_size = user_indices.size(0)

            # Expand the popularity scores for each user in the batch
            return self.normalized_popularity.expand(
                batch_size, -1
            ).clone()  # [batch_size, n_items]

        # Case 'sampled': prediction on a sampled set of items
        return self.normalized_popularity[
            item_indices.clamp(max=self.n_items - 1)
        ]  # [batch_size, pad_seq]
