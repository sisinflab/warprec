# pylint: disable = R0801, E1102
from typing import List, Any, Optional

import torch
from torch import nn, Tensor

from warprec.recommenders.layers import MLP, CNN
from warprec.recommenders.losses import BPRLoss, EmbLoss
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
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The embedding size for users and items.
        cnn_channels (List[int]): The list of output channels for each CNN layer.
        cnn_kernels (List[int]): The list of kernel sizes for each CNN layer.
        cnn_strides (List[int]): The list of stride sizes for each CNN layer.
        dropout_prob (float): The dropout probability for the prediction layer.
        reg_weight (float): The L2 regularization weight.
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
    reg_weight: float
    weight_decay: float
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

        # Check for optional value of block size
        self.block_size = kwargs.get("block_size", 50)

        # Ray Tune converts lists to tuples
        self.cnn_channels = list(self.cnn_channels)
        self.cnn_kernels = list(self.cnn_kernels)
        self.cnn_strides = list(self.cnn_strides)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
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
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_contrastive_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def train_step(self, batch: Any, *args, **kwargs):
        user, pos_item, neg_item = batch

        # Calculate BPR loss
        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)
        bpr_loss = self.bpr_loss(pos_item_score, neg_item_score)

        # Calculate L2 regularization
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        return bpr_loss + reg_loss

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
            for start in range(0, self.n_items, self.block_size):
                end = min(start + self.block_size, self.n_items)
                items_block_indices = torch.arange(start, end)

                # Expand user and item indices to create all pairs for the block
                n_items_in_block = end - start
                users_expanded = (
                    user_indices.unsqueeze(1).expand(-1, n_items_in_block).reshape(-1)
                )
                items_expanded = (
                    items_block_indices.unsqueeze(0).expand(batch_size, -1).reshape(-1)
                ).to(users_expanded.device)

                # Call forward on the flattened batch of pairs for the current block
                scores_flat = self.forward(users_expanded, items_expanded)

                # Reshape the result and append
                scores_block = scores_flat.view(batch_size, n_items_in_block)
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
