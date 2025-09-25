# pylint: disable = R0801, E1102
from typing import Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix

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
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.
    """

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
        self._name = "Random"

    @torch.no_grad()
    def predict_full(
        self,
        user_indices: Tensor,
        train_batch: csr_matrix,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using a random number generator.

        Args:
            user_indices (Tensor): The batch of user indices.
            train_batch (csr_matrix): The batch of train sparse
                interaction matrix.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Generate random predictions
        predictions = torch.rand(train_batch.shape)
        return predictions.to(self._device)

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
        # Generate random predictions
        predictions = torch.rand(item_indices.size())
        return predictions.to(self._device)
