# pylint: disable = R0801, E1102
from typing import Any, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from scipy.sparse import csr_matrix

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.losses import MultiVAELoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z using reparameterization trick."""

    def forward(self, z_mean: Tensor, z_log_var: Tensor) -> Tensor:
        """The forward step of the sampler.

        Args:
            z_mean (Tensor): The mean value.
            z_log_var (Tensor): The log variance value.

        Returns:
            Tensor: The sampled value.
        """
        epsilon = torch.randn_like(z_log_var)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class VAEncoder(nn.Module):
    """Encoder module for MultiVAE with mean and log variance outputs.

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
        self.dense_mean = nn.Linear(intermediate_dim, latent_dim)
        self.dense_log_var = nn.Linear(intermediate_dim, latent_dim)
        self.sampling = Sampling()

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of encoder with normalization, dropout, and sampling.

        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - Tensor: The mean of the inputs.
                - Tensor: The log variance of the inputs.
                - Tensor: The sampled latent vector of the inputs.
        """
        i_normalized = self.l2_normalizer(inputs)
        i_drop = self.dropout(i_normalized)
        x = self.dense_proj(i_drop)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class VADecoder(nn.Module):
    """Decoder module for MultiVAE.

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


@model_registry.register(name="MultiVAE")
class MultiVAE(IterativeRecommender):
    """Implementation of MultiVAE algorithm from
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
        anneal_cap (float): Annealing cap for KL divergence.
        anneal_step (int): Annealing step for KL divergence.
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
    anneal_cap: float
    anneal_step: int

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

        # Encoder with variational components
        self.encoder = VAEncoder(
            original_dim=self.n_items,
            intermediate_dim=self.intermediate_dim,
            latent_dim=self.latent_dim,
            dropout_rate=self.corruption,
        )

        # Decoder
        self.decoder = VADecoder(
            original_dim=self.n_items,
            intermediate_dim=self.intermediate_dim,
            latent_dim=self.latent_dim,
        )

        # Initialize weights
        self.apply(self._init_weights)
        self.loss = MultiVAELoss()

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

    def train_step(self, batch: Any, epoch: int, *args: Any, **kwargs: Any):
        rating_matrix = batch[0]

        anneal = (
            min(self.anneal_cap * epoch / self.anneal_step, self.anneal_cap)
            if self.anneal_step > 0
            else self.anneal_cap
        )
        reconstructed, kl_loss = self.forward(rating_matrix)
        loss: Tensor = self.loss(rating_matrix, reconstructed, kl_loss, anneal)

        return loss

    def forward(self, rating_matrix: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns reconstruction and KL divergence.

        Args:
            rating_matrix (Tensor): The input rating matrix.

        Returns:
            Tuple[Tensor, Tensor]:
                - Tensor: The reconstructed rating matrix.
                - Tensor: The KL divergence loss.
        """
        z_mean, z_log_var, z = self.encoder(rating_matrix)
        reconstructed = self.decoder(z)

        # KL divergence calculation
        kl_loss = -0.5 * torch.mean(
            z_log_var - torch.square(z_mean) - torch.exp(z_log_var) + 1
        )
        return reconstructed, kl_loss

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
                "predict() for MultiVAE requires 'train_batch' as a keyword argument."
            )

        # Compute predictions and convert to Tensor
        train_batch = (
            torch.from_numpy(train_batch_sparse.toarray()).float().to(self.device)
        )
        predictions, _ = self.forward(train_batch)

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
