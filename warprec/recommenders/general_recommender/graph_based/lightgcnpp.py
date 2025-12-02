# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional

import torch
import torch_geometric
import numpy as np
from torch import nn, Tensor
from torch_geometric.nn import LGConv
from torch_sparse import SparseTensor
from scipy.sparse import coo_matrix

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import (
    IterativeRecommender,
)
from warprec.recommenders.general_recommender.graph_based import GraphRecommenderUtils
from warprec.recommenders.losses import BPRLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="LightGCNpp")
class LightGCNpp(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of LightGCN++ algorithm from Revisiting
        LightGCN: Unexpected Inflexibility, Inconsistency, and
        A Remedy Towards Improved Recommendation (RecSys 2024).

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/3640457.3688176>`_.

    Args:
        params (dict): Model parameters. Requires 'alpha', 'beta', 'gamma'.
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
        n_layers (int): The number of graph convolution layers.
        alpha (float): The exponent for the target node degree in the normalization coefficient.
        beta (float): The exponent for the source node degree in the normalization coefficient.
        gamma (float): The coefficient balancing the initial embeddings ($E^0$) and the
            aggregated graph embeddings ($E_{mean}$).
        weight_decay (float): The value of weight decay used in the optimizer.
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
    beta: float
    gamma: float
    weight_decay: float
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

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # LightGCN++ requires custom weights baked into the adjacency matrix,
        # so we must disable LGConv's internal normalization.
        propagation_network_list = []
        for _ in range(self.n_layers):
            propagation_network_list.append(
                (LGConv(normalize=False), "x, edge_index -> x")
            )
        self.propagation_network = torch_geometric.nn.Sequential(
            "x, edge_index", propagation_network_list
        )

        # Compute the adjacency matrix with the weighted version
        self.adj = self._get_weighted_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,  # Adjust for padding idx
        )

        # Init embedding weights
        self.apply(self._init_weights)
        self.loss = BPRLoss()

    def _get_weighted_adj_mat(
        self,
        interaction_matrix: coo_matrix,
        n_users: int,
        n_items: int,
    ) -> SparseTensor:
        """Computes the weighted adjacency matrix based on Alpha and Beta.

        Formula: E_ij = 1 / (deg_i^alpha * deg_j^beta)
        """
        # Extract user and items nodes from interactions
        user_nodes = interaction_matrix.row
        item_nodes = interaction_matrix.col + n_users

        # Efficient degree calculation using numpy
        user_degrees = np.array(interaction_matrix.sum(axis=1)).flatten()
        item_degrees = np.array(interaction_matrix.sum(axis=0)).flatten()

        # Handle zero degrees to avoid division by zero (though unlikely in cleaned data)
        user_degrees[user_degrees == 0] = 1
        item_degrees[item_degrees == 0] = 1

        # Get degrees corresponding to the edges
        d_u = user_degrees[interaction_matrix.row]
        d_i = item_degrees[interaction_matrix.col]

        # Weights for User -> Item edges (Target is Item, Source is User)
        # weight_u2i = 1 / (d_i^alpha * d_u^beta)
        norm_u2i = (d_i**-self.alpha) * (d_u**-self.beta)

        # Weights for Item -> User edges (Target is User, Source is Item)
        # weight_i2u = 1 / (d_u^alpha * d_i^beta)
        norm_i2u = (d_u**-self.alpha) * (d_i**-self.beta)

        # Concatenate rows, cols and values
        row = np.concatenate([user_nodes, item_nodes])
        col = np.concatenate([item_nodes, user_nodes])
        values = np.concatenate([norm_u2i, norm_i2u])

        # Create the edge tensor
        edge_index = torch.tensor(np.vstack([row, col]), dtype=torch.int64)
        edge_weights = torch.tensor(values, dtype=torch.float32)

        # Create SparseTensor with explicit values (weights)
        adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=edge_weights,
            sparse_sizes=(n_users + n_items, n_users + n_items),
        )

        return adj

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

        # Calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        loss: Tensor = self.loss(pos_scores, neg_scores)

        return loss

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Forward pass of LightGCN++ with custom pooling logic"""
        ego_embeddings = self.get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )

        if self.adj.device() != ego_embeddings.device:
            self.adj = self.adj.to(ego_embeddings.device)

        # Store embeddings from each layer
        embeddings_list = [ego_embeddings]

        current_embeddings = ego_embeddings
        for layer_module in self.propagation_network.children():
            # LGConv(normalize=False) will use the weights in self.adj
            current_embeddings = layer_module(current_embeddings, self.adj)
            embeddings_list.append(current_embeddings)

        # LightGCN++ Pooling Strategy
        # Layer 0 (Initial embeddings)
        e_0 = embeddings_list[0]

        # Layers 1 to K
        e_k_list = embeddings_list[1:]

        # Compute mean of layers 1 to K
        # Stack them to compute mean efficiently: [K, N, Emb_Size] -> Mean dim 0
        if len(e_k_list) > 0:
            e_k_mean = torch.stack(e_k_list, dim=0).mean(dim=0)
        else:
            # Fallback if 0 layers (should not happen in GCN)
            e_k_mean = torch.zeros_like(e_0)

        # Combine using Gamma
        final_embeddings = self.gamma * e_0 + (1 - self.gamma) * e_k_mean

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            final_embeddings,
            [self.n_users, self.n_items + 1],
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
            item_embeddings = item_all_embeddings[:-1, :]  # [num_items, embedding_size]
            einsum_string = "be,ie->bi"  # b: batch, e: embedding, i: item
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = item_all_embeddings[
                item_indices
            ]  # [batch_size, pad_seq, embedding_size]
            einsum_string = "be,bse->bs"  # b: batch, e: embedding, s: sample

        predictions = torch.einsum(
            einsum_string, user_embeddings, item_embeddings
        )  # [batch_size, num_items] or [batch_size, pad_seq]
        return predictions
