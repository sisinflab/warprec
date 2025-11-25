# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional

import torch
import torch_geometric
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_normal_
from torch_geometric.nn import LGConv

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import (
    IterativeRecommender,
)
from warprec.recommenders.general_recommender.graph_based import GraphRecommenderUtils
from warprec.recommenders.losses import BPRLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="LightGCN")
class LightGCN(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of LightGCN algorithm from
        LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR 2020)

    For further details, check the `paper <https://arxiv.org/abs/2002.02126>`_.

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
        n_layers (int): The number of graph convolution layers.
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
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,  # Adjust for padding idx
        )

        # Initialization of the propagation network
        propagation_network_list = []
        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(), "x, edge_index -> x"))
        self.propagation_network = torch_geometric.nn.Sequential(
            "x, edge_index", propagation_network_list
        )

        # Vectorized normalization for embedding
        self.alpha = torch.tensor([1 / (k + 1) for k in range(self.n_layers + 1)])

        # Init embedding weights
        self.apply(self._init_weights)
        self.loss = BPRLoss()

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

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
        """Forward pass of the LightGCN model for embedding propagation.

        Returns:
            Tuple[Tensor, Tensor]: User and item embeddings after propagation.
        """
        ego_embeddings = self.get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )
        embeddings_list = [ego_embeddings]

        # This will handle the propagation layer by layer.
        # This is used later to correctly multiply each layer by
        # the corresponding value of alpha
        current_embeddings = ego_embeddings
        for layer_module in self.propagation_network.children():
            current_embeddings = layer_module(current_embeddings, self.adj)
            embeddings_list.append(current_embeddings)

        # Aggregate embeddings using the alpha value
        lightgcn_all_embeddings = torch.zeros_like(ego_embeddings)
        lightgcn_all_embeddings.to(embeddings_list[0].device)  # Move to correct device
        for k, embedding in enumerate(embeddings_list):
            lightgcn_all_embeddings += embedding * self.alpha[k]

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings,
            [self.n_users, self.n_items + 1],  # Adjust for padding idx
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
