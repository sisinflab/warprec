# pylint: disable = R0801, E1102, W0221
from typing import Tuple, Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="DGCF")
class DGCF(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of DGCF algorithm from
    "Disentangled Graph Collaborative Filtering" (SIGIR 2020).

    For further details, check the `paper <https://arxiv.org/abs/2007.01764>`_.

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
        n_factors (int): The number of factors.
        n_iterations (int): The number of routing steps.
        cor_weight (float): The weight of correlation loss.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.

    Raises:
        ValueError: If embedding size is not divisible by n_factors.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    n_factors: int
    n_iterations: int
    cor_weight: float
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
        super().__init__(params, info, seed=seed, *args, **kwargs)

        if self.embedding_size % self.n_factors != 0:
            raise ValueError(
                f"embedding_size ({self.embedding_size}) must be divisible by n_factors ({self.n_factors})."
            )

        self.factor_dim = self.embedding_size // self.n_factors

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # Graph Structure
        # We extract indices once. We do NOT use SparseTensor for propagation
        # because weights change every iteration. Raw scatter/gather is faster.
        self._init_graph_indices(interactions)

        # Init embedding weights
        self.apply(self._init_weights)
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    def _init_graph_indices(self, interactions: Interactions):
        """Prepares graph indices for efficient scatter operations."""
        # Get COO matrix
        adj = interactions.get_sparse().tocoo()

        # Convert to tensors
        row = torch.from_numpy(adj.row).long()
        col = torch.from_numpy(adj.col).long()

        # Create Bi-partite Graph Indices (User->Item AND Item->User)
        # Source Nodes
        src = torch.cat([row, col + self.n_users])
        # Target Nodes
        dst = torch.cat([col + self.n_users, row])

        # Register as buffers (automatically moved to GPU)
        self.register_buffer("edge_index_src", src)
        self.register_buffer("edge_index_dst", dst)

        self.num_nodes = self.n_users + self.n_items + 1
        self.num_edges = src.size(0)

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

    def train_step(self, batch: Any, *args: Any, **kwargs: Any) -> Tensor:
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
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        # Calculate Correlation loss
        cor_loss = self._correlation_loss(u_embeddings, pos_embeddings)
        cor_loss = self.cor_weight * cor_loss

        return bpr_loss + reg_loss + cor_loss

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Performs DGCF propagation using vectorized operations."""

        # Initial Embeddings: [n_nodes, embedding_size]
        ego_embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )

        # Reshape to [n_nodes, n_factors, factor_dim] for vectorized processing
        ego_embeddings = ego_embeddings.view(
            self.num_nodes, self.n_factors, self.factor_dim
        )

        all_embeddings = [ego_embeddings]

        # Initialize Routing Logits (S): [n_edges, n_factors]
        routing_logits = torch.zeros(
            self.num_edges, self.n_factors, device=self.device, dtype=torch.float32
        )

        # Loop over layers
        for _ in range(self.n_layers):
            # Current layer embeddings: [N, K, D]
            current_embeddings = all_embeddings[-1]

            # Loop over iterations (Routing)
            for _ in range(self.n_iterations):
                # Calculate Routing Weights (Softmax over factors)
                routing_weights = F.softmax(routing_logits, dim=1)  # [E, K]

                # Dynamic Normalization
                # Degree of target nodes per factor: [N, K]
                # We sum the incoming edge weights for each node and factor
                deg = torch.zeros(self.num_nodes, self.n_factors, device=self.device)
                deg = deg.index_add(
                    0,
                    self.edge_index_src,  # type: ignore[arg-type]
                    routing_weights,
                )  # Sum rows (src)

                # D^-0.5
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)

                # Norm Edge Weights: [E, K]
                # norm[uv] = D[u]^-0.5 * W[uv] * D[v]^-0.5
                norm_weights = (
                    deg_inv_sqrt[self.edge_index_src]  # type: ignore[index]
                    * routing_weights
                    * deg_inv_sqrt[self.edge_index_dst]  # type: ignore[index]
                )

                # Message Passing
                # Gather Source Embeddings: [E, K, D]
                src_emb = current_embeddings[self.edge_index_src]  # type: ignore[index]

                # Apply Weights: [E, K, D] * [E, K, 1]
                weighted_messages = src_emb * norm_weights.unsqueeze(-1)

                # Aggregate to Target: [N, K, D]
                next_embeddings = torch.zeros_like(current_embeddings)
                next_embeddings = next_embeddings.index_add(
                    0,
                    self.edge_index_dst,  # type: ignore[arg-type]
                    weighted_messages,
                )

                # Gather embeddings for edges: [E, K, D]
                head_emb_edge = next_embeddings[self.edge_index_src]  # type: ignore[index]
                tail_emb_edge = current_embeddings[self.edge_index_dst]  # type: ignore[index]

                # Dot product sum over D -> [E, K]
                delta_logits = (head_emb_edge * torch.tanh(tail_emb_edge)).sum(dim=2)

                routing_logits = routing_logits + delta_logits

            # Update for next layer
            all_embeddings.append(next_embeddings)

        # Sum pooling over layers and factors
        # Stack layers: [L+1, N, K, D]
        # Sum layers: [N, K, D]
        final_embeddings = torch.stack(all_embeddings, dim=0).sum(dim=0)

        # Reshape back to [N, Embed_Size] (Concatenate factors)
        final_embeddings = final_embeddings.view(self.num_nodes, self.embedding_size)

        user_final, item_final = torch.split(
            final_embeddings, [self.n_users, self.n_items + 1]
        )
        return user_final, item_final

    def _distance_correlation(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Calculates Distance Correlation between two tensors."""

        def _centered_distance(x):
            # Pairwise distance matrix: (x-y)^2 = x^2 - 2xy + y^2
            r = torch.sum(x * x, dim=1, keepdim=True)
            d = r - 2 * torch.mm(x, x.t()) + r.t()
            d = torch.sqrt(d.clamp(min=1e-8))

            # Centering
            row_mean = d.mean(dim=1, keepdim=True)
            col_mean = d.mean(dim=0, keepdim=True)
            all_mean = d.mean()
            return d - row_mean - col_mean + all_mean

        def _dist_covariance(d1, d2):
            return (d1 * d2).sum() / (d1.shape[0] ** 2)

        d1 = _centered_distance(x1)
        d2 = _centered_distance(x2)

        dcov_12 = _dist_covariance(d1, d2)
        dcov_11 = _dist_covariance(d1, d1)
        dcov_22 = _dist_covariance(d2, d2)

        return dcov_12 / (torch.sqrt(dcov_11 * dcov_22) + 1e-8)

    def _correlation_loss(self, user_emb: Tensor, item_emb: Tensor) -> Tensor:
        """Computes correlation loss efficiently."""
        # Reshape to [Batch, n_factors, factor_dim]
        # This avoids using torch.chunk which creates list overhead
        user_factors = user_emb.view(-1, self.n_factors, self.factor_dim)
        item_factors = item_emb.view(-1, self.n_factors, self.factor_dim)

        loss = torch.tensor(0.0, device=self.device)

        # We iterate only unique pairs of factors
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                loss = loss + self._distance_correlation(
                    user_factors[:, i], user_factors[:, j]
                )
                loss = loss + self._distance_correlation(
                    item_factors[:, i], item_factors[:, j]
                )

        return loss

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
