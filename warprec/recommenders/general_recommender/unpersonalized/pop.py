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
        interactions (Interactions): The training interactions.
        *args (Any): Argument for PyTorch nn.Module.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Raises:
        ValueError: If the items value was not passed through the info dict.
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
        self.items = info.get("items", None)
        if not self.items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )

        X = interactions.get_sparse()

        # Count the number of items to define the popularity
        popularity = torch.tensor(
            X.sum(axis=0).A1, device=self._device, dtype=torch.float32
        )
        # Count the total number of interactions
        item_count = torch.tensor(X.sum(), device=self._device, dtype=torch.float32)

        # Normalize popularity by the total number of interactions
        # Add epsilon to avoid division by zero if there are no interactions
        self.normalized_popularity = popularity / (item_count + 1e-6)

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
            )  # [batch_size, num_items]

        # Case 'sampled': prediction on a sampled set of items
        return self.normalized_popularity[
            item_indices.clamp(max=self.items - 1)
        ]  # [batch_size, pad_seq]
