from typing import Any, Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_normal_, constant_
from scipy.sparse import csr_matrix
from elliotwo.data.dataset import Interactions
from elliotwo.recommenders.base_recommender import Recommender
from elliotwo.utils.registry import model_registry


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z using reparameterization trick."""

    def forward(self, z_mean: Tensor, z_log_var: Tensor) -> Tensor:
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
class MultiVAE(Recommender):
    """Implementation of MultiVAE algorithm from
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
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        l2_lambda (float): L2 regularization parameter.
        anneal_cap (float): Annealing cap for KL divergence.
        anneal_step (int): Annealing step for KL divergence.
    """

    intermediate_dim: int
    latent_dim: int
    dropout: float
    epochs: int
    learning_rate: float
    l2_lambda: float
    anneal_cap: float
    anneal_step: int

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
        self._name = "MultiVAE"

        # Get dataset information
        self.items = info.get("items")
        if not self.items:
            raise ValueError("Items count must be provided in dataset info")

        # Encoder with variational components
        self.encoder = VAEncoder(
            original_dim=self.items,
            intermediate_dim=self.intermediate_dim,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout,
        )

        # Decoder
        self.decoder = VADecoder(
            original_dim=self.items,
            intermediate_dim=self.intermediate_dim,
            latent_dim=self.latent_dim,
        )

        # Initialize weights
        self.apply(self._init_weights)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda
        )

        # Move to device
        self.to(self._device)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

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

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method.

        The training will be conducted on dense user-item matrix in batches.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        self.train()
        X = interactions.get_sparse()

        for epoch in range(self.epochs):
            epoch_loss = 0.0

            # Annealing schedule for KL divergence
            anneal = (
                min(self.anneal_cap * epoch / self.anneal_step, self.anneal_cap)
                if self.anneal_step > 0
                else self.anneal_cap
            )

            for start in range(0, X.shape[0], interactions.batch_size):
                end = min(start + interactions.batch_size, X.shape[0])

                rating_matrix = torch.tensor(
                    X[start:end].toarray(), device=self._device
                ).float()

                self.optimizer.zero_grad()
                reconstructed, kl_loss = self.forward(rating_matrix)

                # Reconstruction loss
                log_softmax = F.log_softmax(reconstructed, dim=1)
                neg_ll = -torch.mean(torch.sum(log_softmax * rating_matrix, dim=1))

                # Total loss with annealing
                total_loss = neg_ll + anneal * kl_loss

                # Backpropagation
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()

            if report_fn is not None:
                report_fn(self)

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction using the the encoder and decoder modules.

        Args:
            interaction_matrix (csr_matrix): The matrix containing the
                pairs of interactions to evaluate.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Convert to dense matrix
        dense_matrix = torch.tensor(
            interaction_matrix.toarray(), device=self._device
        ).float()

        # Forward pass
        reconstructed, _ = self.forward(dense_matrix)
        log_softmax = F.log_softmax(reconstructed, dim=1)

        # Mask seen items
        seen_mask = torch.tensor(interaction_matrix.toarray() != 0, device=self._device)
        log_softmax[seen_mask] = -torch.inf

        return log_softmax
