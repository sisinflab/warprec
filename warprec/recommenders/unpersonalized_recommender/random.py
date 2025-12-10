# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
from torch import Tensor

from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry


@model_registry.register(name="Random")
class Random(Recommender):
    """Definition of Random unpersonalized model.
    This model will recommend items based on a random number generator.
    """

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
            shape = (batch_size, self.n_items)

            # Generate random scores
            return torch.rand(shape)  # [batch_size, n_items]

        # Case 'sampled': prediction on a sampled set of items
        return torch.rand(item_indices.size())  # [batch_size, pad_seq]
