# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from scipy.sparse import csr_matrix

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="ELSA")
class ELSA(IterativeRecommender):
    """Implementation of ELSA algorithm from
        "Scalable Linear Shallow Autoencoder for Collaborative Filtering" in RecSys 22.

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3523227.3551482>`_.

    Args:
        params (dict): The dictionary with the model params.
        info (dict): The dictionary containing dataset information.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        n_dims (int): The number of dimensions for the latent space.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    DATALOADER_TYPE = DataLoaderType.INTERACTION_LOADER

    n_dims: int
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        # Weight matrix W
        self.W = nn.Parameter(torch.empty([self.n_items, self.n_dims]))
        nn.init.xavier_uniform_(self.W)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_interaction_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def train_step(self, batch: Any, epoch: int, *args, **kwargs) -> Tensor:
        rating_matrix = batch[0]

        # Prediction
        output = self.forward(rating_matrix)

        # Normalized MSE loss
        loss = F.mse_loss(
            F.normalize(output, p=2, dim=-1),
            F.normalize(rating_matrix, p=2, dim=-1),
            reduction="mean",
        )

        return loss

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of ELSA. Computes (xAA^T - x)

        Args:
            x (Tensor): Batch of user-item interactions [batch_size, n_items].

        Returns:
            Tensor: Predicted scores [batch_size, n_items].
        """
        # Normalization of W weight matrix (A)
        A = F.normalize(self.W, p=2, dim=-1)

        # Projection: x @ A (xA)
        latent = torch.matmul(x, A)

        # Reconstruction: latent @ A^T (xAA^T)
        reconstruction = torch.matmul(latent, A.t())

        # Diagonal constraint: x - xAA^T
        return reconstruction - x

    def predict(
        self,
        train_batch: csr_matrix,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Args:
            train_batch (csr_matrix): The batch of user interaction vectors in sparse format.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Convert sparse batch to dense tensor and compute predictions
        x = torch.from_numpy(train_batch.toarray()).to(self.device).float()
        predictions = self.forward(x)

        # Return full or sampled predictions
        if item_indices is None:
            # Case 'full': prediction on all items
            return predictions  # [batch_size, n_items]

        # Case 'sampled': prediction on a sampled set of items
        return predictions.gather(
            1,
            item_indices.to(predictions.device).clamp(
                max=self.n_items - 1
            ),  # [batch_size, pad_seq]
        )
