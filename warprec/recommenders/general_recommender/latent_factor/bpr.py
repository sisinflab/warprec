# pylint: disable = R0801, E1102
from typing import Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_normal_

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.losses import BPRLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="BPR")
class BPR(IterativeRecommender):
    """Implementation of BPR algorithm from
        BPR: Bayesian Personalized Ranking from Implicit Feedback 2012

    For further details, check the `paper <https://arxiv.org/abs/1205.2618>`_.

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
        embedding_size (int): The embedding size of user and item.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Model hyperparameters
    embedding_size: int
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

        # Embeddings
        self.user_embedding = nn.Embedding(users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            items + 1, self.embedding_size, padding_idx=items
        )

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
            xavier_normal_(module.weight.data)

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
        """Forward pass of the BPR model.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.

        Returns:
            Tensor: The predicted score for each pair of positive and negative items.
        """
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)

        return torch.mul(user_e, item_e).sum(dim=1)

    @torch.no_grad()
    def predict_full(
        self,
        user_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the learned embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Retrieve embeddings
        user_e_all = self.user_embedding.weight  # [n_users, embedding_size]
        item_e_all = self.item_embedding.weight[:-1, :]  # [n_items, embedding_size]

        # Select only the embeddings in the current batch
        u_embeddings_batch = user_e_all[user_indices]  # [batch_size, embedding_size]
        predictions = torch.matmul(
            u_embeddings_batch, item_e_all.transpose(0, 1)
        )  # [batch_size, n_items]
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
        # Retrieve embeddings
        # NOTE: .clamp() is used for padded item_indices
        user_embeddings = self.user_embedding(
            user_indices
        )  # [batch_size, embedding_size]
        candidate_item_embeddings = self.item_embedding(
            item_indices
        )  # [batch_size, pad_seq, embedding_size]

        # Compute predictions efficiently
        predictions = torch.einsum(
            "bi,bji->bj", user_embeddings, candidate_item_embeddings
        )
        return predictions.to(self._device)
