# pylint: disable = R0801, E1102
from typing import List, Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import normal_
from warprec.recommenders.layers import MLP
from warprec.data.dataset import Interactions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.registry import model_registry


@model_registry.register(name="NeuMF")
class NeuMF(IterativeRecommender):
    """Implementation of NeuMF algorithm from
        Neural Collaborative Filtering 2017.

    For further details, check the `paper <https://arxiv.org/abs/1708.05031>`_.

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
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, *args, **kwargs)
        self._name = "NeuMF"

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

        # Set block size
        self.block_size = kwargs.get("block_size", 50)

        # Ray Tune converts lists to tuples
        # so we need to convert them back to lists
        self.mlp_hidden_size = list(self.mlp_hidden_size)

        # MF embeddings
        self.user_mf_embedding = nn.Embedding(users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(items, self.mf_embedding_size)

        # MLP embeddings
        self.user_mlp_embedding = nn.Embedding(users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(items, self.mlp_embedding_size)

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

        # Loss and optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        # Move to device
        self.to(self._device)

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def get_optimizer(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def get_loss_function(self):
        return nn.BCEWithLogitsLoss()

    def get_dataloader(self, interactions: Interactions, **kwargs):
        return interactions.get_item_rating_dataloader(num_negatives=self.neg_samples)

    def train_step(self, batch: Any):
        user, item, rating = [x.to(self._device) for x in batch]

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
        train_batch: Tensor,
        user_indices: Tensor,
        user_seq: Tensor,
        seq_len: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the learned embeddings.

        Args:
            train_batch (Tensor): The train batch of user interactions.
            user_indices (Tensor): The batch of user indices.
            user_seq (Tensor): The user sequence of item interactions.
            seq_len (Tensor): The user sequence length.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        batch_size, num_items = train_batch.size()

        preds = []
        for start in range(0, num_items, self.block_size):
            end = min(start + self.block_size, num_items)
            items_block = torch.arange(start, end, device=self._device)  # [block_size]

            # Expand user_indices and items_block to create all user-item pairs
            # within this block.
            users_block = (
                user_indices.unsqueeze(1).expand(-1, end - start).reshape(-1)
            )  # [batch_size * block_size]
            items_block_expanded = (
                items_block.unsqueeze(0).expand(batch_size, -1).reshape(-1)
            )  # [batch_size * block_size]

            preds_block = self.sigmoid(self.forward(users_block, items_block_expanded))
            preds.append(preds_block.view(batch_size, end - start))

        predictions = torch.cat(preds, dim=1)  # [batch_size x num_items]

        # Masking interaction already seen in train
        predictions[train_batch != 0] = -torch.inf
        return predictions.to(self._device)
