# pylint: disable = R0801, E1102
from typing import Any, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor
from scipy.sparse import csr_matrix

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="UltraGCN")
class UltraGCN(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of UltraGCN algorithm from
    "UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation" (CIKM 2021).

    For further details, check the `paper <https://arxiv.org/abs/2110.15114>`_.

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
        w_lambda (float): Weight for the User-Item Constraint Loss (L_C).
        w_gamma (float): Weight for the Item-Item Constraint Loss (L_I).
        w_neg (float): Weight for negative samples in the constraint loss.
        ii_k (int): Number of neighbors (K) for the Item-Item graph construction.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Hyperparameters
    embedding_size: int
    w_lambda: float
    w_gamma: float
    w_neg: float
    ii_k: int
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

        # Initialize weights
        self.apply(self._init_weights)

        # Initialize Regularization Loss
        self.reg_loss = EmbLoss()

        # Pre-compute Constraint Weights (Beta) and Item-Item Graph (Omega)
        self._prepare_constraints(interactions)

    def _prepare_constraints(self, interactions: Interactions):
        """Pre-computes the weights required for the constraint losses."""
        # Get sparse interaction matrix (User x Item)
        R: csr_matrix = interactions.get_sparse()

        # Compute Degrees
        # NOTE: We add +1 to avoid division by zero
        user_degree = np.array(R.sum(axis=1)).squeeze() + 1
        item_degree = np.array(R.sum(axis=0)).squeeze() + 1

        # Optimization: Pre-compute sqrt(degree) to save ops in training loop
        # Beta calculation involves sqrt(d_u) and sqrt(d_i)
        self.register_buffer(
            "user_degree_pow",
            torch.from_numpy(user_degree).pow(-1.0).float().to(self.device),
        )  # 1/d_u
        self.register_buffer(
            "user_sqrt_deg",
            torch.from_numpy(user_degree).sqrt().float().to(self.device),
        )  # sqrt(d_u)
        self.register_buffer(
            "item_sqrt_deg",
            torch.from_numpy(item_degree).sqrt().float().to(self.device),
        )  # sqrt(d_i)

        # Construct Item-Item Graph (L_I)
        # Calculate Co-occurrence: G = R^T * R
        G = R.transpose().dot(R)

        # Set diagonal to 0
        G.setdiag(0)
        G.eliminate_zeros()

        # Calculate Item Degrees in G
        g_degree = np.array(G.sum(axis=1)).squeeze() + 1

        # Select Top-K neighbors for each item
        ii_neighbors = []
        ii_weights = []

        for i in range(self.n_items):
            row_start = G.indptr[i]
            row_end = G.indptr[i + 1]

            if row_end - row_start == 0:
                ii_neighbors.append([0] * self.ii_k)
                ii_weights.append([0.0] * self.ii_k)
                continue

            cols = G.indices[row_start:row_end]
            data = G.data[row_start:row_end]

            if len(data) > self.ii_k:
                top_k_idx = np.argpartition(data, -self.ii_k)[-self.ii_k :]
                cols = cols[top_k_idx]
                data = data[top_k_idx]

            g_i = g_degree[i]
            g_j = g_degree[cols]

            omega = (data / g_i) * np.sqrt(g_i / g_j)

            if len(cols) < self.ii_k:
                pad_len = self.ii_k - len(cols)
                cols = np.pad(
                    cols, (0, pad_len), "constant", constant_values=self.n_items
                )
                omega = np.pad(omega, (0, pad_len), "constant")

            ii_neighbors.append(cols)
            ii_weights.append(omega)

        # Convert lists to arrays before creating the buffers
        ii_neighbors_np = np.array(ii_neighbors)
        ii_weights_np = np.array(ii_weights)

        self.register_buffer(
            "ii_neighbors",
            torch.tensor(ii_neighbors_np, dtype=torch.long).to(self.device),
        )
        self.register_buffer(
            "ii_weights", torch.tensor(ii_weights_np, dtype=torch.float).to(self.device)
        )

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

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Forward pass just returns the embeddings."""
        return self.user_embedding.weight, self.item_embedding.weight

    def train_step(self, batch: Any, epoch: int, *args, **kwargs) -> Tensor:
        user, pos_item, neg_item = batch

        user_emb = self.user_embedding(user)
        pos_item_emb = self.item_embedding(pos_item)
        neg_item_emb = self.item_embedding(neg_item)

        # Calculate scores
        pos_scores = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_scores = torch.mul(user_emb, neg_item_emb).sum(dim=1)

        # Main Objective (L_O)
        # -log(sigmoid(pos)) - log(sigmoid(-neg))
        loss_O = -F.logsigmoid(pos_scores).mean() - F.logsigmoid(-neg_scores).mean()

        # Constraint Loss (L_C)
        # Beta = (1/d_u) * sqrt(d_u) * (1/sqrt(d_i)) = (1/sqrt(d_u)) * (1/sqrt(d_i))

        # Common term for user part of Beta
        beta_user_part = self.user_degree_pow[user] * self.user_sqrt_deg[user]  # type: ignore[index]

        beta_pos = beta_user_part / self.item_sqrt_deg[pos_item]  # type: ignore[index]
        beta_neg = beta_user_part / self.item_sqrt_deg[neg_item]  # type: ignore[index]

        loss_C = (
            -(beta_pos * F.logsigmoid(pos_scores)).mean()
            - (beta_neg * F.logsigmoid(-neg_scores)).mean() * self.w_neg
        )

        # Item-Item Constraint Loss (L_I)
        ii_neighbors = self.ii_neighbors[pos_item]  # type: ignore[index]
        ii_weights = self.ii_weights[pos_item]  # type: ignore[index]
        neighbor_emb = self.item_embedding(ii_neighbors)

        # Dot product [batch_size, 1, embedding_size] * [batch_size, K, embedding_size] -> [batch_size, k]
        ii_scores = torch.mul(pos_item_emb.unsqueeze(1), neighbor_emb).sum(dim=2)
        loss_I = -(ii_weights * F.logsigmoid(ii_scores)).sum(dim=1).mean()

        # Calculate L2 loss
        reg_loss = self.reg_weight * self.reg_loss(user_emb, pos_item_emb, neg_item_emb)

        return loss_O + (self.w_lambda * loss_C) + (self.w_gamma * loss_I) + reg_loss

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
