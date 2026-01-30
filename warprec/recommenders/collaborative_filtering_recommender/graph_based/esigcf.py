# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional

import numpy as np
import torch
from torch import nn, Tensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss, InfoNCELoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="ESIGCF")
class ESIGCF(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of ESIGCF algorithm from
        Extremely Simplified but Intent-enhanced Graph Collaborative Filtering.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The embedding size of items.
        n_layers (int): The number of propagation layers.
        ssl_lambda (float): Weight for SSL Intent Loss (User-Positive).
        can_lambda (float): Weight for Candidate Loss (Positive-Generated).
        temperature (float): Temperature for InfoNCE loss.
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
    ssl_lambda: float
    can_lambda: float
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

        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # Activation functions
        self.activation_layer = nn.Tanh()
        self.gen_activation = nn.LeakyReLU()

        # Initialize weights
        self.apply(self._init_weights)

        # Initialize Losses
        self.bpr_loss_func = BPRLoss()
        self.reg_loss_func = EmbLoss()
        self.nce_loss = InfoNCELoss(temperature=self.temperature)

        # Precompute Adjacency Matrices with Hybrid Normalization
        self._init_graphs(interactions)

    def _init_graphs(self, interactions: Interactions):
        """Constructs the specific adjacency matrices required for JoGCN with Hybrid Normalization.

        Hybrid Norm: Val = 1/Du + 1/sqrt(Du*Di)

        Registers two buffers:
            1. user_graph (R): [n_users, n_items] - For initial user embedding generation.
            2. full_graph (A): [n_users + n_items, n_users + n_items] - For propagation.
        """
        # Get Interaction Matrix (User x Item)
        R = interactions.get_sparse().tocoo()
        row = R.row
        col = R.col

        # Calculate Degrees
        user_degree = np.array(R.sum(axis=1)).squeeze()
        item_degree = np.array(R.sum(axis=0)).squeeze()

        # Handle division by zero
        user_degree[user_degree == 0] = 1.0
        item_degree[item_degree == 0] = 1.0

        # Calculate Hybrid Values
        # Norm = (1 / |Nu|) + (1 / sqrt(|Nu||Ni|))
        d_u_inv = 1.0 / user_degree
        d_u_sqrt = np.sqrt(user_degree)
        d_i_sqrt = np.sqrt(item_degree)

        # Values for edges (u, i)
        val_mean = d_u_inv[row]
        val_sym = 1.0 / (d_u_sqrt[row] * d_i_sqrt[col])
        val_hybrid = val_mean + val_sym

        # Construct user_graph (R)
        indices = np.vstack((row, col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(val_hybrid)
        shape = (self.n_users, self.n_items + 1)  # +1 for padding consistency

        user_graph = torch.sparse_coo_tensor(i, v, shape).coalesce()
        self.register_buffer("user_graph", user_graph)

        # Construct full_graph (A)
        # Top-Right (User -> Item): Hybrid Norm
        val_tr = val_hybrid

        # Bottom-Left (Item -> User): Hybrid Norm
        # Mean part: 1/Di
        # Sym part: 1/sqrt(Di*Du)
        d_i_inv = 1.0 / item_degree
        val_bl_mean = d_i_inv[col]
        val_bl_sym = 1.0 / (d_i_sqrt[col] * d_u_sqrt[row])
        val_bl = val_bl_mean + val_bl_sym

        # Concatenate for full matrix
        # Rows: [row, col + n_users]
        # Cols: [col + n_users, row]
        full_row = np.concatenate([row, col + self.n_users])
        full_col = np.concatenate([col + self.n_users, row])
        full_val = np.concatenate([val_tr, val_bl])

        indices_full = np.vstack((full_row, full_col))
        i_full = torch.LongTensor(indices_full)
        v_full = torch.FloatTensor(full_val)

        # Size: Users + Items + 1 (padding)
        total_nodes = self.n_users + self.n_items + 1
        shape_full = (total_nodes, total_nodes)

        full_graph = torch.sparse_coo_tensor(i_full, v_full, shape_full).coalesce()
        self.register_buffer("full_graph", full_graph)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs: Any,
    ):
        return interactions.get_pos_neg_dataloader(
            batch_size=self.batch_size, low_memory=low_memory
        )

    def train_step(self, batch: Any, epoch: int, *args, **kwargs) -> Tensor:
        user, pos_item, neg_item = batch

        # Get propagated embeddings
        all_user_embeddings, all_item_embeddings = self.forward()

        # Get embeddings for current batch users and items
        u_emb = all_user_embeddings[user]
        pos_emb = all_item_embeddings[pos_item]
        neg_emb = all_item_embeddings[neg_item]

        # Get initial embeddings for regularization
        ego_pos_emb = self.item_embedding(pos_item)
        ego_neg_emb = self.item_embedding(neg_item)

        # Generate "Intent-aware" Negative Item
        # Generated Negative = LeakyReLU(Pos * Neg)
        can_neg_emb = self.gen_activation(pos_emb * neg_emb)

        # Calculate BPR loss
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        bpr_loss = self.bpr_loss_func(pos_scores, neg_scores)

        # Calculate L2 loss
        reg_loss = self.reg_weight * self.reg_loss_func(ego_pos_emb, ego_neg_emb)

        # SSL Intent Loss (User - Positive Item)
        # Align user and positive item
        ssl_intent_loss = self.nce_loss(u_emb, pos_emb)
        ssl_loss = self.ssl_lambda * ssl_intent_loss

        # Candidate/Item Loss (Positive Item - Generated Negative)
        # Align Positive Item with Generated Negative (Intent View)
        can_item_loss = self.nce_loss(pos_emb, can_neg_emb)
        can_loss = self.can_lambda * can_item_loss

        return bpr_loss + reg_loss + ssl_loss + can_loss

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Forward pass of ESIGCF.

        Steps performed:
            1. Construct initial user embeddings from item embeddings using user_graph.
            2. Concatenate User and Item embeddings.
            3. Propagate through Full Graph with Tanh activation.
            4. Sum representations from all layers.
        """
        item_embedding = self.item_embedding.weight

        # Initial User Embedding = Activation(R * Item_Emb)
        user_embedding_0 = self.activation_layer(
            torch.sparse.mm(self.user_graph, item_embedding)  # [n_users, n_items+1]
        )

        # Concatenate for propagation: [n_users + n_items + 1, dim]
        all_embedding = torch.cat([user_embedding_0, item_embedding], dim=0)

        all_embeddings_list = [all_embedding]

        # Iterative Propagation
        for _ in range(self.n_layers):
            all_embedding = torch.sparse.mm(self.full_graph, all_embedding)
            all_embedding = self.activation_layer(all_embedding)
            all_embeddings_list.append(all_embedding)

        # Aggregation: Summation
        final_all_embeddings = torch.stack(all_embeddings_list, dim=1).sum(dim=1)

        # Split back to User and Item
        users_emb, items_emb = torch.split(
            final_all_embeddings, [self.n_users, self.n_items + 1]
        )

        return users_emb, items_emb

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
        user_all_embeddings, item_all_embeddings = self.forward()

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
