# pylint: disable = R0801, E1102
from typing import List, Any, Optional

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import normal_

from warprec.recommenders.layers import MLP, CNN
from warprec.recommenders.losses import BPRLoss
from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.enums import DataLoaderType
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
        DATALOADER_TYPE: The type of dataloader used.
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

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

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

        # Get information from dataset info
        self.users = info.get("users", None)
        if not self.users:
            raise ValueError(
                "Users value must be provided to correctly initialize the model."
            )
        self.items = info.get("items", None)
        if not self.items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        self.block_size = kwargs.get("block_size", 50)

        # Ray Tune converts lists to tuples
        self.cnn_channels = list(self.cnn_channels)
        self.cnn_kernels = list(self.cnn_kernels)
        self.cnn_strides = list(self.cnn_strides)

        self.user_embedding = nn.Embedding(self.users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.items + 1, self.embedding_size, padding_idx=self.items
        )
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

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return interactions.get_pos_neg_dataloader(
            batch_size=self.batch_size, low_memory=low_memory
        )

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
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the learned embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Retrieve batch size from user batch
        batch_size = user_indices.size(0)

        if item_indices is None:
            # Case 'full': iterate through all items in memory-safe blocks
            all_scores = []
            for start in range(0, self.items, self.block_size):
                end = min(start + self.block_size, self.items)
                items_block_indices = torch.arange(start, end, device=self._device)

                # Expand user and item indices to create all pairs for the block
                num_items_in_block = end - start
                users_expanded = (
                    user_indices.unsqueeze(1).expand(-1, num_items_in_block).reshape(-1)
                )
                items_expanded = (
                    items_block_indices.unsqueeze(0).expand(batch_size, -1).reshape(-1)
                )

                # Call forward on the flattened batch of pairs for the current block
                scores_flat = self.forward(users_expanded, items_expanded)

                # Reshape the result and append
                scores_block = scores_flat.view(batch_size, num_items_in_block)
                all_scores.append(scores_block)

            # Concatenate the results from all blocks
            predictions = torch.cat(all_scores, dim=1)
            return predictions

        # Case 'sampled': process all given item_indices at once
        pad_seq = item_indices.size(1)

        # Expand user and item indices to create all pairs
        users_expanded = user_indices.unsqueeze(1).expand(-1, pad_seq).reshape(-1)
        items_expanded = item_indices.reshape(-1)

        # Call forward on the flattened batch of pairs
        predictions_flat = self.forward(users_expanded, items_expanded)

        # Reshape the flat predictions back to the original batch shape
        predictions = predictions_flat.view(batch_size, pad_seq)
        return predictions
