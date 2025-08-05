from typing import Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_normal_, constant_
from warprec.data.dataset import Interactions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.losses import MultiDAELoss
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
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items value was not passed through the info dict.

    Attributes:
        intermediate_dim (int): Intermediate dimension size.
        latent_dim (int): Latent dimension size.
        dropout (float): Dropout probability.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    intermediate_dim: int
    latent_dim: int
    dropout: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, *args, **kwargs)
        self._name = "MultiDAE"

        # Get dataset information
        self.items = info.get("items")
        if not self.items:
            raise ValueError("Items count must be provided in dataset info")

        # Encoder layers
        self.encoder = Encoder(
            original_dim=self.items,
            intermediate_dim=self.intermediate_dim,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout,
        )

        # Decoder layers
        self.decoder = Decoder(
            original_dim=self.items,
            intermediate_dim=self.intermediate_dim,
            latent_dim=self.latent_dim,
        )

        # Initialize weights
        self.apply(self._init_weights)
        self.loss = MultiDAELoss()

        # Move to device
        self.to(self._device)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def get_dataloader(self, interactions: Interactions, **kwargs):
        return interactions.get_interaction_loader(batch_size=self.batch_size)

    def train_step(self, batch: Any, *args: Any, **kwargs: Any):
        rating_matrix = [x.to(self._device) for x in batch][0]

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
        h = F.dropout(h, self.dropout, training=self.training)

        # Encode and decode
        h = self.encoder(h)
        return self.decoder(h)

    @torch.no_grad()
    def predict(
        self,
        train_batch: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the the encoder and decoder modules.

        Args:
            train_batch (Tensor): The train batch of user interactions.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Forward pass
        predictions = self.forward(train_batch)

        # Masking interaction already seen in train
        predictions[train_batch != 0] = -torch.inf
        return predictions.to(self._device)
