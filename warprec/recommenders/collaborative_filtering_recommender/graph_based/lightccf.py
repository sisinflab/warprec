# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional

import torch
from torch import nn, Tensor
from torch_geometric.nn import LGConv

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss, InfoNCELoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="LightCCF")
class LightCCF(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of LightCCF algorithm from
    "Unveiling Contrastive Learning’s Capability of Neighborhood Aggregation for Collaborative Filtering" (SIGIR 2025).

    LightCCF introduces a Neighborhood Aggregation (NA) loss that brings users closer to
    all their interacted items while pushing them away from other positive pairs.
    It can operate with a simple Base Encoder (MF) or a GCN Encoder.

    For further details, please refer to the `paper <https://arxiv.org/abs/2504.10113>`_.

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
        n_layers (int): The number of graph convolution layers. If 0, uses Base Encoder (MF).
        alpha (float): The weight for the Neighborhood Aggregation.
        temperature (float): The temperature coefficient for InfoNCE.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    alpha: float
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

        # Graph Initialization (Only if n_layers > 0)
        if self.n_layers > 0:
            self.adj = self.get_adj_mat(
                interactions.get_sparse().tocoo(),
                self.n_users,
                self.n_items + 1,  # Adjust for padding idx
            )
            self.propagation_network = nn.ModuleList(
                [LGConv() for _ in range(self.n_layers)]
            )
        else:
            self.adj = None
            self.propagation_network = None

        # Initialize weights
        self.apply(self._init_weights)

        # Initialize Losses
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.nce_loss = InfoNCELoss(temperature=self.temperature)

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

    def train_step(self, batch: Any, *args, **kwargs) -> Tensor:
        user, pos_item, neg_item = batch

        # Get propagated embeddings
        user_all_embeddings, item_all_embeddings = self.forward()

        # Get embeddings for current batch users and items
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # Calculate BPR loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        # Calculate InfoNCE loss
        combined_view = torch.cat([pos_embeddings, u_embeddings], dim=0)
        na_loss = self.alpha * self.nce_loss(u_embeddings, combined_view)

        # Calculate L2 loss
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        return bpr_loss + na_loss + reg_loss

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        If n_layers > 0, performs LightGCN-style aggregation.
        If n_layers == 0, returns raw embeddings (MF encoder).

        Returns:
            Tuple[Tensor, Tensor]: User and Item embeddings.
        """
        # Base embeddings
        ego_user_emb = self.user_embedding.weight
        ego_item_emb = self.item_embedding.weight

        # Graph propagation (if enabled)
        if self.n_layers > 0:
            all_embeddings = torch.cat([ego_user_emb, ego_item_emb], dim=0)

            if self.adj.device() != all_embeddings.device:
                self.adj = self.adj.to(all_embeddings.device)

            embeddings_list = [all_embeddings]
            current_embeddings = all_embeddings

            for conv_layer in self.propagation_network:
                current_embeddings = conv_layer(current_embeddings, self.adj)
                embeddings_list.append(current_embeddings)

            final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)

            user_final, item_final = torch.split(
                final_embeddings, [self.n_users, self.n_items + 1]
            )
            return user_final, item_final

        # Base Encoder (MF)
        return ego_user_emb, ego_item_emb

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
