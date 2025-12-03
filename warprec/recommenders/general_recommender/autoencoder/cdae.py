# pylint: disable = R0801, E1102, W0221
from typing import Any, Optional

import torch
from torch import nn, Tensor
from scipy.sparse import csr_matrix

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="CDAE")
class CDAE(IterativeRecommender):
    """Implementation of CDAE algorithm from
    "Collaborative Denoising Auto-Encoders for Top-N Recommender Systems." in WSDM 2016.

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/2835776.2835837>`_.

    This model learns latent representations by training a denoising autoencoder on corrupted
    user-item interaction vectors, incorporating a user-specific embedding to guide the
    reconstruction.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items or users value was not passed through the info dict.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The dimension of the user embeddings and the hidden layer.
        corruption (float): The probability of dropout applied to the input layer (denoising).
        hid_activation (str): The activation function for the hidden layer ('relu', 'tanh', 'sigmoid').
        out_activation (str): The activation function for the output layer ('relu', 'sigmoid').
        loss_type (str): The loss function to use for backpropagation ('bce', 'mse').
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used during training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.INTERACTION_LOADER_WITH_USER_ID

    # Model hyperparameters
    embedding_size: int
    corruption: float
    hid_activation: str
    out_activation: str
    loss_type: str
    reg_weight: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        *args: Any,
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, seed=seed, *args, **kwargs)

        self.n_users = info.get("users", None)
        if not self.n_users:
            raise ValueError("Users value must be provided.")
        self.n_items = info.get("items", None)
        if not self.n_items:
            raise ValueError("Items value must be provided.")

        # User-specific embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)

        # Encoder for the item interaction vector
        self.item_encoder = nn.Linear(self.n_items, self.embedding_size)

        # Decoder to reconstruct the interaction vector
        self.decoder = nn.Linear(self.embedding_size, self.n_items)

        # Dropout layer for the "denoising" aspect
        self.corruption_dropout = nn.Dropout(p=self.corruption)

        # Activation functions
        self.h_act = self._get_activation(self.hid_activation)
        self.o_act = self._get_activation(self.out_activation)

        # Define loss type to use
        self.main_loss: nn.Module
        if self.loss_type == "MSE":
            self.main_loss = nn.MSELoss()
        elif self.loss_type == "BCE":
            self.main_loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Invalid loss_type, loss_type must be in [MSE, BCE]")
        self.reg_loss = EmbLoss()

        # Initialize weights
        self.apply(self._init_weights)

    def _get_activation(self, name: str) -> nn.Module:
        """Helper to get an activation function module from its name."""
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "sigmoid":
            return nn.Sigmoid()
        raise ValueError(f"Invalid activation function name: {name}")

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return interactions.get_interaction_loader(
            batch_size=self.batch_size,
            include_user_id=True,
            low_memory=low_memory,
        )

    def forward(self, user_history: Tensor, user_indices: Tensor) -> Tensor:
        """Performs the forward pass of the CDAE model.

        Args:
            user_history (Tensor): The user-item interaction vector [batch_size, num_items].
            user_indices (Tensor): The user indices for the batch [batch_size].

        Returns:
            Tensor: The reconstructed interaction vector (logits) [batch_size, num_items].
        """
        # Encode the item history
        item_hidden = self.item_encoder(user_history)

        # Get the user-specific bias
        user_bias = self.user_embedding(user_indices)

        # Combine them (the "Collaborative" part)
        combined = self.h_act(item_hidden + user_bias)

        # Decode to reconstruct the original vector
        reconstructed_logits = self.decoder(combined)

        return reconstructed_logits

    def train_step(self, batch: Any, *args: Any, **kwargs: Any):
        user_indices, user_history = batch

        # Apply corruption ("Denoising" part)
        corrupted_history = self.corruption_dropout(user_history)

        # Get the reconstructed logits from the corrupted input
        reconstructed_logits = self.forward(corrupted_history, user_indices)

        # Calculate the reconstruction loss against the original history
        if self.loss_type == "MSE":
            prediction = self.o_act(reconstructed_logits)
            main_loss = self.main_loss(prediction, user_history)
        else:
            main_loss = self.main_loss(reconstructed_logits, user_history)

        # Calculate L2 regularization
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user_indices),
        )

        return main_loss + reg_loss

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
        predictions_logits = self.forward(train_batch, user_indices)

        # Apply the final activation function to get scores
        predictions: Tensor = self.o_act(predictions_logits)

        if item_indices is None:
            # Case 'full': prediction on all items
            return predictions  # [batch_size, num_items]

        # Case 'sampled': prediction on a sampled set of items
        return predictions.gather(
            1,
            item_indices.to(predictions.device).clamp(
                max=self.n_items - 1
            ),  # [batch_size, pad_seq]
        )
