# pylint: disable = R0801, E1102
from typing import List, Any, Optional

import torch
from torch import nn, Tensor

from warprec.recommenders.layers import MLP
from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="NeuMF")
class NeuMF(IterativeRecommender):
    """Implementation of NeuMF algorithm from
        Neural Collaborative Filtering 2017.

    For further details, check the `paper <https://arxiv.org/abs/1708.05031>`_.

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
        mf_embedding_size (int): The MF embedding size.
        mlp_embedding_size (int): The MLP embedding size.
        mlp_hidden_size (List[int]): The MLP hidden layer size list.
        mf_train (bool): Wether or not to train MF embedding.
        mlp_train (bool): Wether or not to train MLP embedding.
        dropout (float): The dropout probability.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples per positive interaction.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.ITEM_RATING_LOADER

    # Model hyperparameters
    mf_embedding_size: int
    mlp_embedding_size: int
    mlp_hidden_size: List[int]
    mf_train: bool
    mlp_train: bool
    dropout: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float
    neg_samples: int

    def __init__(
        self,
        params: dict,
        *args: Any,
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, seed=seed, *args, **kwargs)

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

        # Set block size
        self.block_size = kwargs.get("block_size", 50)

        # Ray Tune converts lists to tuples
        # so we need to convert them back to lists
        self.mlp_hidden_size = list(self.mlp_hidden_size)

        # MF embeddings
        self.user_mf_embedding = nn.Embedding(self.users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(
            self.items + 1, self.mf_embedding_size, padding_idx=self.items
        )

        # MLP embeddings
        self.user_mlp_embedding = nn.Embedding(self.users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(
            self.items + 1, self.mlp_embedding_size, padding_idx=self.items
        )

        # MLP layers
        self.mlp_layers = MLP(
            [2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout
        )

        # Final prediction layer
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(
                self.mf_embedding_size + self.mlp_hidden_size[-1], 1
            )
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        else:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        # Init embedding weights
        self.apply(self._init_weights)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return interactions.get_item_rating_dataloader(
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
            low_memory=low_memory,
        )

    def train_step(self, batch: Any, *args, **kwargs):
        user, item, rating = batch

        predictions = self(user, item)
        loss: Tensor = self.loss(predictions, rating)

        return loss

    # pylint: disable = E0606
    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Forward pass of the NeuMF model.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.

        Returns:
            Tensor: The predicted score for each pair (user, item).
        """
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        output: Tensor = None

        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)

        if self.mlp_train:
            mlp_input = torch.cat((user_mlp_e, item_mlp_e), -1)
            mlp_output = self.mlp_layers(mlp_input)

        if self.mf_train and self.mlp_train:
            combined = torch.cat((mf_output, mlp_output), -1)
            output = self.predict_layer(combined)
        elif self.mf_train:
            output = self.predict_layer(mf_output)
        else:
            output = self.predict_layer(mlp_output)

        return output.squeeze(-1)

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
            preds_logits = []
            for start in range(0, self.items, self.block_size):
                end = min(start + self.block_size, self.items)
                items_block = torch.arange(start, end)

                # Expand user and item indices to create all pairs for the block
                users_block = (
                    user_indices.unsqueeze(1).expand(-1, end - start).reshape(-1)
                )
                items_block_expanded = (
                    items_block.unsqueeze(0).expand(batch_size, -1).reshape(-1)
                ).to(users_block.device)

                # Get raw logits from the forward pass
                logits_block = self.forward(users_block, items_block_expanded)
                preds_logits.append(logits_block.view(batch_size, -1))

            predictions_logits = torch.cat(preds_logits, dim=1)

        else:
            # Case 'sampled': process all given item_indices at once
            pad_seq = item_indices.size(1)

            # Expand user and item indices to create all pairs
            users_expanded = user_indices.unsqueeze(1).expand(-1, pad_seq).reshape(-1)
            items_expanded = item_indices.reshape(-1)

            # Get raw logits from the forward pass
            predictions_flat_logits = self.forward(users_expanded, items_expanded)
            predictions_logits = predictions_flat_logits.view(batch_size, pad_seq)

        # Apply sigmoid once to the final logits tensor
        predictions = self.sigmoid(predictions_logits)
        return predictions
