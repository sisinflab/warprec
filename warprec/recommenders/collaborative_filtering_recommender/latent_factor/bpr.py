# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
from torch import nn, Tensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="BPR")
class BPR(IterativeRecommender):
    """Implementation of BPR algorithm from
        BPR: Bayesian Personalized Ranking from Implicit Feedback 2012

    For further details, please refer to the `paper <https://arxiv.org/abs/1205.2618>`_.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The embedding size of user and item.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Model hyperparameters
    embedding_size: int
    reg_weight: float
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

        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

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

        # Compute BPR loss
        pos_item_score = self.forward(user, pos_item)
        neg_item_score = self.forward(user, neg_item)
        bpr_loss = self.bpr_loss(pos_item_score, neg_item_score)

        # Compute L2 regularization
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        return bpr_loss + reg_loss

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
        # Retrieve user embeddings
        user_embeddings = self.user_embedding(
            user_indices
        )  # [batch_size, embedding_size]

        if item_indices is None:
            # Case 'full': prediction on all items
            item_embeddings = self.item_embedding.weight[
                :-1, :
            ]  # [n_items, embedding_size]
            einsum_string = "be,ie->bi"  # b: batch, e: embedding, i: item
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = self.item_embedding(
                item_indices
            )  # [batch_size, pad_seq, embedding_size]
            einsum_string = "be,bse->bs"  # b: batch, e: embedding, s: sample

        # Common prediction step
        predictions = torch.einsum(
            einsum_string, user_embeddings, item_embeddings
        )  # [batch_size, n_items] or [batch_size, pad_seq]
        return predictions
