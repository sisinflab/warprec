# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional

import torch
from torch import nn, Tensor
from torch_sparse import SparseTensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss, InfoNCELoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="SGL")
class SGL(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of SGL algorithm from
        Self-supervised Graph Learning for Recommendation (SIGIR 2021)

    For further details, check the `paper <https://arxiv.org/abs/2010.10783>`_.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The embedding size of user and item.
        n_layers (int): The number of graph convolution layers.
        ssl_tau (float): The temperature parameter for SSL loss.
        ssl_reg (float): The weight for SSL loss.
        dropout (float): The dropout rate for graph augmentation.
        aug_type (str): The type of graph augmentation ('ED', 'ND', 'RW').
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.

    Raises:
        ValueError: If aug_type is not one of the supported types.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Hyperparameters
    embedding_size: int
    n_layers: int
    ssl_tau: float
    ssl_reg: float
    dropout: float
    aug_type: str
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, info, *args, seed=seed, **kwargs)

        # Validation
        self.aug_type = self.aug_type.upper()
        if self.aug_type not in ["ED", "ND", "RW"]:
            raise ValueError(
                f"Invalid aug_type: {self.aug_type}. "
                "Supported types: 'ED' (Edge Dropout), 'ND' (Node Dropout), 'RW' (Random Walk)."
            )

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # Graph Construction
        # We keep the original adjacency matrix for the main task
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,
            normalize=True,
        )

        # Pre-fetch COO representation for fast augmentation.
        # We cache the indices on the device for the augmentation methods
        row, col, _ = self.adj.coo()
        self.adj_row = row
        self.adj_col = col
        self.adj_size = self.adj.sparse_sizes()

        # Initialize weights
        self.apply(self._init_weights)

        # Losses
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.nce_loss = InfoNCELoss(temperature=self.ssl_tau)

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

    def _graph_augmentation(self) -> SparseTensor:
        """Fast generation of Edge Dropout view using cached indices."""
        if self.dropout == 0:
            return self.adj

        # Generate mask on the device directly
        num_edges = self.adj_row.size(0)
        keep_mask = torch.rand(num_edges, device=self.device) > self.dropout

        # Apply mask to cached indices
        new_row = self.adj_row[keep_mask]
        new_col = self.adj_col[keep_mask]

        # Apply mask to values
        _, _, vals = self.adj.coo()
        new_val = vals[keep_mask]

        # Create temporary SparseTensor before normalization
        temp_adj = SparseTensor(
            row=new_row,
            col=new_col,
            value=new_val,
            sparse_sizes=self.adj_size,
            is_sorted=True,
        )

        # Normalize the new adjacency matrix
        row_sum = temp_adj.sum(dim=1)  # D
        d_inv_sqrt = row_sum.pow(-0.5)  # D^-0.5
        d_inv_sqrt.masked_fill_(
            d_inv_sqrt == float("inf"), 0.0
        )  # Handle division by zero

        # Get the normalization values
        d_mat_rows = d_inv_sqrt[new_row]
        d_mat_cols = d_inv_sqrt[new_col]
        new_norm_vals = d_mat_rows * d_mat_cols

        # Fast construction using is_sorted=True
        return SparseTensor(
            row=new_row,
            col=new_col,
            value=new_norm_vals,
            sparse_sizes=self.adj_size,
            is_sorted=True,
        )

    def _node_dropout_mask(self, num_nodes: int) -> Tensor:
        """Generates a mask for Node Dropout (ND)."""
        if self.dropout == 0:
            return torch.ones(num_nodes, 1, device=self.device)
        return (torch.rand(num_nodes, 1, device=self.device) > self.dropout).float()

    def forward(
        self, adj: SparseTensor, augment: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with optional augmentation logic."""
        ego_u = self.user_embedding.weight
        ego_i = self.item_embedding.weight
        ego_all = torch.cat([ego_u, ego_i], dim=0)

        # Node Dropout (ND) applied to features
        if augment and self.aug_type == "ND":
            mask = self._node_dropout_mask(ego_all.size(0))
            ego_all = ego_all * mask

        embeddings_list = [ego_all]

        # Ensure adj is on device
        if adj.device() != self.device:
            adj = adj.to(self.device)

        curr_adj = adj

        for _ in range(self.n_layers):
            # Random Walk (RW): New graph structure at EACH layer
            if augment and self.aug_type == "RW":
                curr_adj = self._graph_augmentation()

            # Propagation
            next_emb = curr_adj.matmul(embeddings_list[-1])
            embeddings_list.append(next_emb)

        # Aggregation (Mean)
        final_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)

        user_final, item_final = torch.split(
            final_embeddings, [self.n_users, self.n_items + 1]
        )
        return user_final, item_final

    def train_step(self, batch: Any, *args, **kwargs) -> Tensor:
        user, pos_item, neg_item = batch

        # Get propagated embeddings
        user_all_embeddings, item_all_embeddings = self.forward(self.adj, augment=False)

        # Get embeddings for current batch users and items
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # Calculate BPR loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        # Calculate SSL loss
        ssl_loss = torch.tensor(0.0, device=self.device)

        if self.ssl_reg > 0:
            # View 1 Generation
            if self.aug_type == "ED":
                adj_v1 = self._graph_augmentation()
                user_v1, item_v1 = self.forward(adj_v1, augment=False)
            else:
                # ND and RW handle augmentation internally
                user_v1, item_v1 = self.forward(self.adj, augment=True)

            # View 2 Generation
            if self.aug_type == "ED":
                adj_v2 = self._graph_augmentation()
                user_v2, item_v2 = self.forward(adj_v2, augment=False)
            else:
                user_v2, item_v2 = self.forward(self.adj, augment=True)

            # Calculate InfoNCE loss
            # Users
            loss_u = self.nce_loss(user_v1[user], user_v2[user])

            # Items (Positive items only)
            loss_i = self.nce_loss(item_v1[pos_item], item_v2[pos_item])

            ssl_loss = self.ssl_reg * (loss_u + loss_i)

        # Calculate L2 loss
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        return bpr_loss + ssl_loss + reg_loss

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
        user_all_embeddings, item_all_embeddings = self.forward(self.adj)

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
