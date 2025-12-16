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


@model_registry.register(name="EGCF")
class EGCF(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of EGCF algorithm from
    "Simplify to the Limit! Embedding-Less Graph Collaborative Filtering for Recommender Systems" (TOIS 2024).

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/3701230>`_.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The embedding size of items.
        n_layers (int): The number of propagation layers.
        ssl_lambda (float): Weight for the Self-Supervised Learning (InfoNCE) loss.
        temperature (float): Temperature parameter for InfoNCE loss.
        mode (str): Propagation mode, either 'parallel' or 'alternating'.
        reg_weight (float): Weight for the regularization loss.
        weight_decay (float): The weight decay (L2 regularization).
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.

    Raises:
        ValueError: If the mode is not supported.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    ssl_lambda: float
    temperature: float
    mode: str
    reg_weight: float
    weight_decay: float
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

        if self.mode not in ["parallel", "alternating"]:
            raise ValueError(
                f"Unsupported mode: {self.mode}. Available modes are 'parallel' and 'alternating'."
            )

        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # User-Item Interaction Matrix (R)
        self.R_adj = self._get_user_item_adj(interactions)

        # Bipartite Adjacency Matrix (A)
        if self.mode == "parallel":
            self.bipartite_adj = self.get_adj_mat(
                interactions.get_sparse().tocoo(),
                self.n_users,
                self.n_items + 1,  # Adjust for padding idx
                normalize=True,
            )

        self.activation = nn.Tanh()

        # Init embedding weights
        self.apply(self._init_weights)
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.nce_loss = InfoNCELoss(self.temperature)

    def _get_user_item_adj(self, interactions: Interactions) -> Tensor:
        """Constructs the normalized User-Item interaction matrix R.
        Formula: R_norm = D_u^(-1/2) * R * D_i^(-1/2)

        Args:
            interactions (Interactions): The interaction object.

        Returns:
            Tensor: The sparse normalized adjacency matrix (User x Item).
        """
        # Get sparse interaction matrix in COO format
        adj = interactions.get_sparse().tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)

        # Compute degrees for normalization
        # Degree of user u = number of items they interacted with
        # Degree of item i = number of users who interacted with it
        deg_user = torch.zeros(self.n_users, dtype=torch.float32)

        # Use n_items + 1 to be safe with dimensions, though we only populate up to n_items
        deg_item = torch.zeros(self.n_items + 1, dtype=torch.float32)

        deg_user.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        deg_item.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float32))

        # Inverse sqrt degree
        deg_inv_sqrt_user = deg_user.pow(-0.5)
        deg_inv_sqrt_user.masked_fill_(deg_inv_sqrt_user == float("inf"), 0.0)

        deg_inv_sqrt_item = deg_item.pow(-0.5)
        deg_inv_sqrt_item.masked_fill_(deg_inv_sqrt_item == float("inf"), 0.0)

        # Values for the sparse matrix: 1 / sqrt(Du * Di)
        values = deg_inv_sqrt_user[row] * deg_inv_sqrt_item[col]

        # Create sparse tensor
        indices = torch.stack([row, col], dim=0)
        shape = (self.n_users, self.n_items + 1)

        # We use coalesce to ensure indices are sorted and unique
        R_adj = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        return R_adj

    def _sparse_mm(self, sparse_mat: Any, dense_mat: Tensor) -> Tensor:
        """Wrapper to handle both torch.Tensor (sparse) and torch_sparse.SparseTensor
        matrix multiplication.
        """
        if isinstance(sparse_mat, Tensor):
            # Standard PyTorch sparse tensor
            return torch.sparse.mm(sparse_mat, dense_mat)
        elif isinstance(sparse_mat, SparseTensor):
            # torch_sparse.SparseTensor
            return sparse_mat.matmul(dense_mat)
        else:
            raise TypeError(f"Unsupported sparse matrix type: {type(sparse_mat)}")

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
        user_all_embeddings, item_all_embeddings = self.forward()

        # Get embeddings for current batch users and items
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # Calculate BPR loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        # Calculate L2 loss
        reg_loss = self.reg_weight * self.reg_loss(
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        # Calculate InfoNCE loss
        ssl_user_loss = self.nce_loss(
            u_embeddings, u_embeddings
        )  # User-User Uniformity (L_user)
        ssl_pos_loss = self.nce_loss(
            pos_embeddings, pos_embeddings
        )  # Item-Item Uniformity (L_item)
        ssl_inter_loss = self.nce_loss(
            u_embeddings, pos_embeddings
        )  # User-Item Alignment (L_inter)
        ssl_loss = self.ssl_lambda * (ssl_user_loss + ssl_pos_loss + ssl_inter_loss)

        return bpr_loss + reg_loss + ssl_loss

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Forward pass of EGCF.

        Generates user and item embeddings based on the selected mode.

        Returns:
            Tuple[Tensor, Tensor]: The final user and item embeddings.
        """
        # Ensure R_adj is on the correct device
        if self.R_adj.device != self.device:
            self.R_adj = self.R_adj.to(self.device)

        if self.mode == "parallel":
            return self._forward_parallel()
        else:
            return self._forward_alternating()

    def _forward_alternating(self) -> Tuple[Tensor, Tensor]:
        """Alternating Iteration.

        Propagates information back and forth between users and items using R and R.T.
        """
        # Initial Item Embedding
        item_emb = self.item_embedding.weight

        all_user_embeddings = []
        all_item_embeddings = []

        current_item_emb = item_emb

        for _ in range(self.n_layers):
            # User = Tanh(R * Item)
            user_emb = self.activation(self._sparse_mm(self.R_adj, current_item_emb))

            # Item = Tanh(R.T * User)
            item_emb_next = self.activation(self._sparse_mm(self.R_adj.t(), user_emb))

            all_user_embeddings.append(user_emb)
            all_item_embeddings.append(item_emb_next)

            current_item_emb = item_emb_next

        # Sum pooling
        final_user_embeddings = torch.stack(all_user_embeddings, dim=1).sum(dim=1)
        final_item_embeddings = torch.stack(all_item_embeddings, dim=1).sum(dim=1)

        return final_user_embeddings, final_item_embeddings

    def _forward_parallel(self) -> Tuple[Tensor, Tensor]:
        """Parallel Iteration.

        Performs the following steps:
            1. Construct initial User embedding from Items.
            2. Concatenate and propagate using standard bipartite graph.
        """
        # Ensure bipartite_adj is on the correct device
        if self.bipartite_adj.device() != self.device:
            self.bipartite_adj = self.bipartite_adj.to(self.device)

        item_emb_0 = self.item_embedding.weight

        # Construct initial user embedding: e_u^(0) = R * e_i^(0)
        user_emb_0 = self._sparse_mm(self.R_adj, item_emb_0)

        # Concatenate [User, Item]
        all_embedding = torch.cat([user_emb_0, item_emb_0], dim=0)

        embeddings_list = []

        # Propagate using Bipartite Graph
        current_embedding = all_embedding
        for _ in range(self.n_layers):
            # A * E
            current_embedding = self._sparse_mm(self.bipartite_adj, current_embedding)
            # Activation
            current_embedding = self.activation(current_embedding)
            embeddings_list.append(current_embedding)

        # Sum pooling
        final_embeddings = torch.stack(embeddings_list, dim=1).sum(dim=1)

        # Split back to User and Item
        user_all_embeddings, item_all_embeddings = torch.split(
            final_embeddings, [self.n_users, self.n_items + 1]
        )

        return user_all_embeddings, item_all_embeddings

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
