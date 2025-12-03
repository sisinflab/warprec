# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional

import torch
import numpy as np
import scipy.sparse as sp
from torch import nn, Tensor
from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import (
    IterativeRecommender,
)
from warprec.recommenders.general_recommender.graph_based import (
    GraphRecommenderUtils,
    SparseDropout,
    NGCFLayer,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="NGCF")
class NGCF(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of NGCF algorithm from
        Neural Graph Collaborative Filtering (SIGIR 2019)

    For further details, check the `paper <https://arxiv.org/abs/1905.08166>`_.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items or users value was not passed through the info dict.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The embedding size of user and item.
        weight_size (list[int]): List of hidden sizes for each layer.
        node_dropout (float): Dropout rate for nodes in the adjacency matrix.
        message_dropout (float): Dropout rate for messages/embeddings during propagation.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Model hyperparameters
    embedding_size: int
    weight_size: list[int]
    node_dropout: float
    message_dropout: float
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, seed=seed, *args, **kwargs)

        # Get information from dataset info
        self.n_users = info.get("users", None)
        if not self.n_users:
            raise ValueError(
                "Users value must be provided to correctly initialize the model."
            )
        self.n_items = info.get("items", None)
        if not self.n_items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )

        # Initialize the hidden dimensions
        self.hidden_size_list = [
            self.embedding_size
        ] + self.weight_size  # [embed_k, layer1_dim, layer2_dim, ...]

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )
        self.adj = self._get_norm_adj_mat_ngcf(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,  # Adjust for padding idx
        )

        # Optionally define a dropout layer (optimized for sparse data)
        self.sparse_dropout = (
            SparseDropout(self.node_dropout) if self.node_dropout > 0 else None
        )

        # Initialization of the propagation network
        self.propagation_network = nn.ModuleList()
        for i in range(len(self.weight_size)):
            in_f = self.hidden_size_list[i]
            out_f = self.hidden_size_list[i + 1]
            self.propagation_network.append(
                NGCFLayer(in_f, out_f, self.message_dropout)
            )

        # Init embedding weights
        self.apply(self._init_weights)
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

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
        brp_loss = self.bpr_loss(pos_scores, neg_scores)

        # Calculate L2 regularization
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        return brp_loss + reg_loss

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Forward pass of the NGCF model for embedding propagation.

        Returns:
            Tuple[Tensor, Tensor]: User and item embeddings after propagation.
        """
        # Get the ego_embeddings [user + item]
        ego_embeddings = self.get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )

        # Ensure adjacency matrix is on the same device as embeddings
        if self.adj.device != ego_embeddings.device:
            self.adj = self.adj.to(ego_embeddings.device)

        embeddings_list = [ego_embeddings]

        # Apply dropout if required from hyperparameters
        adj_matrix_current = self.adj
        if self.sparse_dropout is not None:
            adj_matrix_current = self.sparse_dropout(self.adj)

        # Forward each embedding through the sequential
        # propagation network
        current_embeddings = ego_embeddings
        for layer_module in self.propagation_network:
            current_embeddings = layer_module(current_embeddings, adj_matrix_current)
            embeddings_list.append(current_embeddings)

        # Concatenate embeddings from all layers (including ego-embeddings)
        # along the feature dimension
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings,
            [self.n_users, self.n_items + 1],  # Adjust for padding idx
        )
        return user_all_embeddings, item_all_embeddings

    def _get_norm_adj_mat_ngcf(
        self,
        interaction_matrix: coo_matrix,
        n_users: int,
        n_items: int,
    ) -> SparseTensor:
        """Get the normalized interaction matrix of users and items specific to NGCF.
        This includes constructing the full adjacency matrix and applying symmetric normalization.

        Args:
            interaction_matrix (coo_matrix): The full interaction matrix in coo format.
            n_users (int): The number of users.
            n_items (int): The number of items.

        Returns:
            SparseTensor: The sparse normalized adjacency matrix (A_hat).
        """
        # Build adjacency matrix (A)
        # [num_user + num_items x num_user + num_items]
        A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()

        # Add user-item interactions
        for u, i in zip(inter_M.row, inter_M.col):
            A[u, i + n_users] = 1.0  # user -> item
        # Add item-user interactions (transpose)
        for i, u in zip(inter_M_t.row, inter_M_t.col):
            A[i + n_users, u] = 1.0  # item -> user

        A = (
            A.tocsr()
        )  # Convert to CSR for efficient row-wise sum and diagonal matrix creation

        # Symmetric Normalization: D^{-0.5} A D^{-0.5}
        sum_rows = np.array(A.sum(axis=1)).flatten()
        # Add epsilon to avoid division by zero
        sum_rows[sum_rows == 0] = 1e-7
        diag_inv_sqrt = np.power(sum_rows, -0.5)
        D_inv_sqrt = sp.diags(diag_inv_sqrt)

        # L = D^{-0.5} A D^{-0.5}
        L = D_inv_sqrt.dot(A).dot(D_inv_sqrt)

        # Convert to COO format for SparseTensor
        L_coo = L.tocoo()
        indices = torch.LongTensor(np.vstack((L_coo.row, L_coo.col)))
        values = torch.FloatTensor(L_coo.data)
        shape = torch.Size(L_coo.shape)

        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

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
            item_embeddings = item_all_embeddings[:-1, :]  # [num_items, embedding_size]
            einsum_string = "be,ie->bi"  # b: batch, e: embedding, i: item
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = item_all_embeddings[
                item_indices
            ]  # [batch_size, pad_seq, embedding_size]
            einsum_string = "be,bse->bs"  # b: batch, e: embedding, s: sample

        # Compute scores using the appropriate einsum operation
        predictions = torch.einsum(
            einsum_string, user_embeddings, item_embeddings
        )  # [batch_size, num_items] or [batch_size, pad_seq]
        return predictions
