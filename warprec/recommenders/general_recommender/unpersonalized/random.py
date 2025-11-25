# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
from torch import Tensor

from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import model_registry


@model_registry.register(name="Random")
class Random(Recommender):
    """Definition of Random unpersonalized model.
    This model will recommend items based on a random number generator,
    ensuring that previously seen items are not recommended again.

    Args:
        params (dict): The dictionary with the model params.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Raises:
        ValueError: If the items value was not passed through the info dict.
    """

    def __init__(
        self,
        params: dict,
        *args: Any,
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, seed=seed, info=info, *args, **kwargs)
        self.items = info.get("items", None)
        if not self.items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )

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
            shape = (batch_size, self.items)

            # Generate random scores
            return torch.rand(shape)  # [batch_size, num_items]

        # Case 'sampled': prediction on a sampled set of items
        return torch.rand(item_indices.size())  # [batch_size, pad_seq]
