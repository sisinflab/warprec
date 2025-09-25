# pylint: disable = R0801, E1102
from typing import List, Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import normal_
from scipy.sparse import csr_matrix

from warprec.recommenders.layers import MLP, CNN
from warprec.recommenders.losses import BPRLoss
from warprec.data.dataset import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.registry import model_registry


@model_registry.register(name="ConvNCF")
class ConvNCF(IterativeRecommender):
    """Implementation of ConvNCF algorithm from
        Outer Product-based Neural Collaborative Filtering 2018.

    For further details, check the `paper <https://arxiv.org/abs/1808.03912>`_.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items or users value was not passed through the info dict.

    Attributes:
        embedding_size (int): The embedding size for users and items.
        cnn_channels (List[int]): The list of output channels for each CNN layer.
        cnn_kernels (List[int]): The list of kernel sizes for each CNN layer.
        cnn_strides (List[int]): The list of stride sizes for each CNN layer.
        dropout_prob (float): The dropout probability for the prediction layer.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Model hyperparameters
    embedding_size: int
    cnn_channels: List[int]
    cnn_kernels: List[int]
    cnn_strides: List[int]
    dropout_prob: float
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
        self._name = "ConvNCF"

        # Get information from dataset info
        users = info.get("users", None)
        if not users:
            raise ValueError(
                "Users value must be provided to correctly initialize the model."
            )
        items = info.get("items", None)
        if not items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        self.block_size = kwargs.get("block_size", 50)

        # Ray Tune converts lists to tuples
        self.cnn_channels = list(self.cnn_channels)
        self.cnn_kernels = list(self.cnn_kernels)
        self.cnn_strides = list(self.cnn_strides)

        self.user_embedding = nn.Embedding(users, self.embedding_size)
        self.item_embedding = nn.Embedding(items, self.embedding_size)
        self.cnn_layers = CNN(
            self.cnn_channels,
            self.cnn_kernels,
            self.cnn_strides,
            activation="relu",
        )

        # Prediction layer (MLP)
        # The input of the prediction layer is the output
        # of the CNN, so self.cnn_channels[-1]
        self.predict_layers = MLP(
            [self.cnn_channels[-1], 1], self.dropout_prob, activation=None
        )  # We set no activation for last layer

        # Init embedding weights
        self.apply(self._init_weights)
        self.loss = BPRLoss()

        # Move to device
        self.to(self._device)

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def get_dataloader(self, interactions: Interactions, sessions: Sessions, **kwargs):
        return interactions.get_pos_neg_dataloader(self.batch_size)

    def train_step(self, batch: Any, *args, **kwargs):
        user, pos_item, neg_item = [x.to(self._device) for x in batch]

        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)
        loss: Tensor = self.loss(pos_item_score, neg_item_score)

        return loss

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Forward pass of the ConvNCF model.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.

        Returns:
            Tensor: The predicted score for each pair (user, item).
        """
        user_e = self.user_embedding(user)  # [batch_size, embedding_size]
        item_e = self.item_embedding(item)  # [batch_size, embedding_size]

        # Outer product to create interaction map
        interaction_map = torch.bmm(
            user_e.unsqueeze(2), item_e.unsqueeze(1)
        )  # [batch_size, embedding_size, embedding_size]

        # Add a channel dimension for CNN input: [batch_size, 1, embedding_size, embedding_size]
        interaction_map = interaction_map.unsqueeze(1)

        # CNN layers
        cnn_output = self.cnn_layers(
            interaction_map
        )  # [batch_size, cnn_channels[-1], H', W']

        # Sum across spatial dimensions (H', W')
        # This reduces the feature map to [batch_size, cnn_channels[-1]]
        cnn_output = cnn_output.sum(axis=(2, 3))

        # Prediction layer (MLP)
        prediction = self.predict_layers(cnn_output)  # [batch_size, 1]

        return prediction.squeeze(-1)  # [batch_size]

    @torch.no_grad()
    def predict_full(
        self,
        user_indices: Tensor,
        train_batch: csr_matrix,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Prediction using the learned embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            train_batch (csr_matrix): The batch of train sparse
                interaction matrix.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        batch_size, num_items = train_batch.shape
        user_e_batch = self.user_embedding(user_indices)

        all_scores = []
        # We must iterate over items in blocks due to memory constraints
        for item_start_idx in range(0, num_items, self.block_size):
            item_end_idx = min(item_start_idx + self.block_size, num_items)

            item_indices_block = torch.arange(
                item_start_idx, item_end_idx, device=self._device
            )
            item_e_block = self.item_embedding(item_indices_block)
            num_items_in_block = len(item_indices_block)

            # Expand embeddings to create all user-item pairs in the block
            user_e_exp = user_e_batch.unsqueeze(1).expand(-1, num_items_in_block, -1)
            item_e_exp = item_e_block.unsqueeze(0).expand(batch_size, -1, -1)

            # Compute outer product for the entire block
            interaction_map = torch.bmm(
                user_e_exp.reshape(-1, self.embedding_size, 1),
                item_e_exp.reshape(-1, 1, self.embedding_size),
            )
            interaction_map = interaction_map.unsqueeze(1)

            # Pass through layers
            cnn_output = self.cnn_layers(interaction_map)
            cnn_output = cnn_output.sum(axis=(2, 3))
            prediction = self.predict_layers(cnn_output)

            # Reshape scores to [num_users_in_batch, num_items_in_block]
            scores_block = prediction.reshape(batch_size, num_items_in_block)
            all_scores.append(scores_block)

        predictions = torch.cat(all_scores, dim=1)
        return predictions.to(self._device)

    @torch.no_grad()
    def predict_sampled(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction of given items using the learned embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            item_indices (Tensor): The batch of item indices.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x pad_seq}.
        """
        batch_size, pad_seq = item_indices.size()

        # Prepare user and item indices for a single forward pass
        # This flattens the tensors to create a list of all user-item pairs
        # to be evaluated.
        users_expanded = user_indices.unsqueeze(1).expand(-1, pad_seq).reshape(-1)

        # Clamp item indices to handle padding (-1)
        items_expanded = item_indices.clamp(min=0).reshape(-1)

        # Use the forward pass to compute scores for all pairs at once.
        # This is a more concise and reusable way to get the predictions.
        predictions_flat = self.forward(users_expanded, items_expanded)

        # Reshape the flat predictions back to the original batch shape
        predictions = predictions_flat.view(batch_size, pad_seq)
        return predictions.to(self._device)
