from typing import Any, Optional, Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_normal_, constant_
from scipy.sparse import csr_matrix
from elliotwo.data.dataset import Interactions
from elliotwo.recommenders.base_recommender import Recommender
from elliotwo.utils.registry import model_registry


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
class MultiDAE(Recommender):
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
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        l2_lambda (float): L2 regularization parameter.
    """

    intermediate_dim: int
    latent_dim: int
    dropout: float
    epochs: int
    learning_rate: float
    l2_lambda: float

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

        for _ in range(self.epochs):
            epoch_loss = 0.0
            for start in range(0, interactions.batch_size, interactions.batch_size):
                end = min(start + interactions.batch_size, X.shape[0])

                # Create rating matrix
                rating_matrix = torch.tensor(
                    X[start:end].toarray(), device=self._device
                ).float()

                # Forward pass
                self.optimizer.zero_grad()
                reconstructed = self.forward(rating_matrix)

                # Cross-entropy loss
                loss = -(F.log_softmax(reconstructed, 1) * rating_matrix).sum(1).mean()

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if report_fn is not None:
                report_fn(self, loss=epoch_loss)

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
        predictions = self.forward(dense_matrix)

        # Mask seen items
        seen_mask = torch.tensor(interaction_matrix.toarray() != 0, device=self._device)
        predictions[seen_mask] = -torch.inf

        return predictions
