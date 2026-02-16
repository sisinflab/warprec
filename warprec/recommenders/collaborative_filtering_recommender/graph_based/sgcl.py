# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="SGCL")
class SGCL(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of SGCL from
        "SGCL: Unifying Self-Supervised and Supervised Learning for Graph Recommendation" (RecSys '25).

    SGCL unifies the recommendation and contrastive learning tasks into a single
    supervised contrastive loss function. It eliminates the need for data augmentation,
    negative sampling, and multi-task optimization.

    For further details, please refer to the `paper <https://arxiv.org/abs/2507.13336>`_.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The embedding size of user and item.
        n_layers (int): The number of graph convolution layers.
        temperature (float): The temperature parameter for the contrastive loss.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.ITEM_RATING_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    temperature: float
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # Graph Construction (LightGCN style)
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,
            normalize=True,
        )

        self.apply(self._init_weights)
        self.reg_loss = EmbLoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_pointwise_dataloader(
            neg_samples=0,
            batch_size=self.batch_size,
            **kwargs,
        )

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Forward pass of SGCL (Standard LightGCN propagation)."""
        ego_u = self.user_embedding.weight
        ego_i = self.item_embedding.weight
        ego_all = torch.cat([ego_u, ego_i], dim=0)

        embeddings_list = [ego_all]

        if self.adj.device() != self.device:
            self.adj = self.adj.to(self.device)

        for _ in range(self.n_layers):
            next_emb = self.adj.matmul(embeddings_list[-1])
            embeddings_list.append(next_emb)

        final_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)
        user_final, item_final = torch.split(
            final_embeddings, [self.n_users, self.n_items + 1]
        )

        return user_final, item_final

    def train_step(self, batch: Any, *args, **kwargs) -> Tensor:
        user, pos_item, _ = batch  # Ignore rating for SGCL

        # Get embeddings from propagation network
        user_all, item_all = self.forward()

        u_emb = user_all[user]  # [Batch, Dim]
        i_emb = item_all[pos_item]  # [Batch, Dim]

        # Normalize embeddings
        u_emb_norm = F.normalize(u_emb, dim=1)
        i_emb_norm = F.normalize(i_emb, dim=1)

        # Compute SGCL loss
        # L = - log ( exp(pos) / (sum(exp(uu)) + sum(exp(vv))) )
        #   = - pos + log(sum(exp(uu)) + sum(exp(vv)))

        # Numerator (Positive alignment)
        # (u * i) / tau
        pos_scores = (u_emb_norm * i_emb_norm).sum(dim=1) / self.temperature

        # Denominator (Batch Uniformity)
        u_sim_matrix = (
            torch.mm(u_emb_norm, u_emb_norm.t()) / self.temperature
        )  # [Batch, Batch]
        i_sim_matrix = (
            torch.mm(i_emb_norm, i_emb_norm.t()) / self.temperature
        )  # [Batch, Batch]
        all_sims = torch.cat([u_sim_matrix, i_sim_matrix], dim=1)
        log_denominator = torch.logsumexp(all_sims, dim=1)

        # Loss = - (Alignment - Uniformity)
        sgcl_loss = -(pos_scores - log_denominator).mean()

        # Calculate L2 loss
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user), self.item_embedding(pos_item)
        )

        return sgcl_loss + reg_loss

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
        # Retrieve all user and item embeddings from the propagation network
        user_all_embeddings, item_all_embeddings = self.propagate_embeddings()

        # Get the embeddings for the specific users in the batch
        user_embeddings = user_all_embeddings[
            user_indices
        ]  # [batch_size, embedding_size]

        if item_indices is None:
            # Case 'full': prediction on all items
            item_embeddings = item_all_embeddings[:-1, :]  # [n_items, embedding_size]
            einsum_string = "be,ie->bi"  # b: batch, e: embedding, i: item
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = item_all_embeddings[
                item_indices
            ]  # [batch_size, pad_seq, embedding_size]
            einsum_string = "be,bse->bs"  # b: batch, e: embedding, s: sample

        predictions = torch.einsum(
            einsum_string, user_embeddings, item_embeddings
        )  # [batch_size, n_items] or [batch_size, pad_seq]
        return predictions
