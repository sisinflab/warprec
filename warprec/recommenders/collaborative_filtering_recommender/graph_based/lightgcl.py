# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional, List

import numpy as np
import scipy.sparse as sp
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


@model_registry.register(name="LightGCL")
class LightGCL(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of LightGCL algorithm from
    "LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation" (ICLR 2023).

    LightGCL utilizes Singular Value Decomposition (SVD) to construct a global
    contrastive view, which is contrasted with the local graph view (GCN) to
    enhance representation learning.

    For further details, check the `paper <https://arxiv.org/abs/2302.08191>`_.

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
        q (int): The rank for SVD approximation.
        ssl_lambda (float): Weight for contrastive loss.
        temperature (float): Temperature for InfoNCE.
        dropout (float): Dropout probability for the adjacency matrix.
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
    q: int
    ssl_lambda: float
    temperature: float
    dropout: float
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

        # Graph Construction (Normalized Adjacency)
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,  # Adjust for padding idx
            normalize=True,
        )

        # SVD Decomposition (Pre-computed)
        # We perform SVD on the normalized adjacency matrix
        self._perform_svd(interactions)

        # Initialize weights
        self.apply(self._init_weights)

        # Losses
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.info_nce_loss = InfoNCELoss(temperature=self.temperature)

    def _perform_svd(self, interactions: Interactions):
        """Performs Truncated SVD on the normalized adjacency matrix.

        Constructs:
            U_s = U * S
            V_s = V * S
            U_t = U^T
            V_t = V^T
        """

        # Get normalized adjacency matrix as scipy sparse matrix
        R = interactions.get_sparse().tocoo()
        n_nodes = self.n_users + self.n_items

        # Construct the bipartite adjacency matrix
        row = np.concatenate([R.row, R.col + self.n_users])
        col = np.concatenate([R.col + self.n_users, R.row])
        data = np.ones(len(row))

        adj = sp.coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        # Normalize: D^-0.5 * A * D^-0.5
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)

        # Perform SVD
        # U: [N, q], S: [q], Vt: [q, N]
        u, s, _ = sp.linalg.svds(norm_adj, k=self.q)

        # Handle negative strides from scipy svds
        u = u.copy()

        # U_mul_s = U * S
        u_mul_s = u @ np.diag(s)

        # Padding for the last item (padding idx)
        pad_row = np.zeros((1, self.q), dtype=u.dtype)
        u = np.vstack([u, pad_row])
        u_mul_s = np.vstack([u_mul_s, pad_row])

        # Convert to tensors and register buffers
        # These are dense matrices of shape [N_nodes, q]
        self.register_buffer("svd_u", torch.from_numpy(u).float().to(self.device))
        self.register_buffer(
            "svd_u_mul_s", torch.from_numpy(u_mul_s).float().to(self.device)
        )

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

    def _dropout_adj(self, adj: SparseTensor) -> SparseTensor:
        """Applies edge dropout to the torch_sparse.SparseTensor."""
        if self.training and self.dropout > 0:
            # Extract values from SparseTensor
            _, _, val = adj.coo()

            # Create dropout mask
            mask = torch.rand(val.size(0), device=val.device) > self.dropout

            # Apply mask and scale
            # We cast mask to float to perform multiplication
            val = val * mask.to(val.dtype) / (1.0 - self.dropout)

            # Return a new SparseTensor with updated values
            # set_value is efficient as it reuses the index storage
            return adj.set_value(val, layout="coo")

        return adj

    def forward(self) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        """Forward pass computing both GCN and SVD views."""
        # Initial Embeddings
        ego_u = self.user_embedding.weight
        ego_i = self.item_embedding.weight
        ego_all = torch.cat([ego_u, ego_i], dim=0)

        # Lists to store layer embeddings
        gcn_embeddings = [ego_all]
        svd_embeddings = [ego_all]

        # Ensure adj is on device
        if self.adj.device() != self.device:
            self.adj = self.adj.to(self.device)

        # Apply dropout for GCN view
        adj_dropped = self._dropout_adj(self.adj)

        for _ in range(self.n_layers):
            # GCN Propagation
            # Z^(l) = A * Z^(l-1)
            # We use the previous layer's GCN embedding
            z_prev = gcn_embeddings[-1]
            z_next = adj_dropped.matmul(z_prev)
            gcn_embeddings.append(z_next)

            # Project to latent space: [q, N] @ [N, d] -> [q, d]
            latent = torch.mm(self.svd_u.t(), z_prev)  # type: ignore[operator]

            # Reconstruct: [N, q] @ [q, d] -> [N, d]
            g_next = torch.mm(self.svd_u_mul_s, latent)  # type: ignore[arg-type]

            svd_embeddings.append(g_next)

        # Sum pooling
        gcn_final = torch.stack(gcn_embeddings, dim=0).sum(dim=0)

        # Split for BPR compute
        user_gcn, item_gcn = torch.split(gcn_final, [self.n_users, self.n_items + 1])

        return user_gcn, item_gcn, gcn_embeddings, svd_embeddings

    def train_step(self, batch: Any, *args, **kwargs) -> Tensor:
        user, pos_item, neg_item = batch

        # Forward Pass
        user_gcn, item_gcn, gcn_list, svd_list = self.forward()

        # Calculate BPR loss (on GCN view)
        u_e = user_gcn[user]
        pos_e = item_gcn[pos_item]
        neg_e = item_gcn[neg_item]

        pos_scores = torch.mul(u_e, pos_e).sum(dim=1)
        neg_scores = torch.mul(u_e, neg_e).sum(dim=1)
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        # Contrastive Learning (Layer-wise InfoNCE)
        cl_loss = 0.0
        for i in range(self.n_layers + 1):
            # Split current layer embeddings
            u_gcn_l, i_gcn_l = torch.split(
                gcn_list[i], [self.n_users, self.n_items + 1]
            )
            u_svd_l, i_svd_l = torch.split(
                svd_list[i], [self.n_users, self.n_items + 1]
            )

            # Users Contrast
            cl_loss += self.info_nce_loss(u_gcn_l[user], u_svd_l[user])

            # Items Contrast (Only positive items usually)
            cl_loss += self.info_nce_loss(i_gcn_l[pos_item], i_svd_l[pos_item])

        cl_loss = self.ssl_lambda * cl_loss

        # Calculate L2 loss
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        return bpr_loss + cl_loss + reg_loss

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
        # Retrieve all user and item gcn embeddings
        user_gcn, item_gcn, _, _ = self.forward()

        # Get the embeddings for the specific users in the batch
        user_embeddings = user_gcn[user_indices]  # [batch_size, embedding_size]

        if item_indices is None:
            # Case 'full': prediction on all items
            item_embeddings = item_gcn[:-1, :]  # [n_items, embedding_size]
            einsum_string = "be,ie->bi"  # b: batch, e: embedding, i: item
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = item_gcn[
                item_indices
            ]  # [batch_size, pad_seq, embedding_size]
            einsum_string = "be,bse->bs"  # b: batch, e: embedding, s: sample

        predictions = torch.einsum(
            einsum_string, user_embeddings, item_embeddings
        )  # [batch_size, n_items] or [batch_size, pad_seq]
        return predictions
