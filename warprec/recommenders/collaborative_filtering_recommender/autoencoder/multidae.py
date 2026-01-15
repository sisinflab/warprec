# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from scipy.sparse import csr_matrix

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.losses import MultiDAELoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class Encoder(nn.Module):
    """Encoder module for MultiDAE model.

    Args:
        original_dim (int): The original dimension of the input.
        intermediate_dim (int): The intermediate dimension size.
        latent_dim (int): The latent dimension size.
        dropout_rate (float): The dropout probability.
    """

    def __init__(
        self,
        original_dim: int,
        intermediate_dim: int,
        latent_dim: int,
        dropout_rate: float = 0,
    ):
        super().__init__()

        self.l2_normalizer = lambda x: F.normalize(x, p=2, dim=1)
        self.dropout = nn.Dropout(dropout_rate)

        self.dense_proj = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim), nn.Tanh()
        )
        self.dense_mean = nn.Sequential(
            nn.Linear(intermediate_dim, latent_dim), nn.Tanh()
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass of encoder with normalization and dropout."""
        i_normalized = self.l2_normalizer(inputs)
        i_drop = self.dropout(i_normalized)
        x = self.dense_proj(i_drop)
        return self.dense_mean(x)


class Decoder(nn.Module):
    """Decoder module for MultiDAE model.

    Args:
        original_dim (int): The original dimension of the input.
        intermediate_dim (int): The intermediate dimension size.
        latent_dim (int): The latent dimension size.
    """

    def __init__(self, original_dim: int, intermediate_dim: int, latent_dim: int):
        super().__init__()

        self.dense_proj = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim), nn.Tanh()
        )
        self.dense_output = nn.Linear(intermediate_dim, original_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass of decoder."""
        x = self.dense_proj(inputs)
        return self.dense_output(x)


@model_registry.register(name="MultiDAE")
class MultiDAE(IterativeRecommender):
    """Implementation of MultiDAE algorithm from
        Variational Autoencoders for Collaborative Filtering 2018.

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/3178876.3186150>`_.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        intermediate_dim (int): Intermediate dimension size.
        latent_dim (int): Latent dimension size.
        corruption (float): The probability of dropout applied to the input layer (denoising).
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.INTERACTION_LOADER

    intermediate_dim: int
    latent_dim: int
    corruption: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, info, *args, seed=seed, **kwargs)

        # Encoder layers
        self.encoder = Encoder(
            original_dim=self.n_items,
            intermediate_dim=self.intermediate_dim,
            latent_dim=self.latent_dim,
            dropout_rate=self.corruption,
        )

        # Decoder layers
        self.decoder = Decoder(
            original_dim=self.n_items,
            intermediate_dim=self.intermediate_dim,
            latent_dim=self.latent_dim,
        )

        # Initialize weights
        self.apply(self._init_weights)
        self.loss = MultiDAELoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return interactions.get_interaction_loader(
            batch_size=self.batch_size, low_memory=low_memory
        )

    def train_step(self, batch: Any, *args: Any, **kwargs: Any):
        rating_matrix = batch[0]

        reconstructed = self(rating_matrix)
        loss: Tensor = self.loss(rating_matrix, reconstructed)

        return loss

    def forward(self, rating_matrix: Tensor) -> Tensor:
        """Forward pass with normalization and dropout.

        Args:
            rating_matrix (Tensor): The input rating matrix.

        Returns:
            Tensor: The reconstructed rating matrix.
        """
        # Normalize input
        h = F.normalize(rating_matrix, dim=1)

        # Apply dropout
        h = F.dropout(h, self.corruption, training=self.training)

        # Encode and decode
        h = self.encoder(h)
        return self.decoder(h)

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the the encoder and decoder modules.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.

        Raises:
            ValueError: If the 'train_batch' keyword argument is not provided.
        """
        # Get train batch from kwargs
        train_batch_sparse: Optional[csr_matrix] = kwargs.get("train_batch")
        if train_batch_sparse is None:
            raise ValueError(
                "predict() for MultiDAE requires 'train_batch' as a keyword argument."
            )

        # Compute predictions and convert to Tensor
        train_batch = (
            torch.from_numpy(train_batch_sparse.toarray()).float().to(self.device)
        )
        predictions = self.forward(train_batch)

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
