# pylint: disable = R0801, E1102, W0221
from typing import List, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from torch_sparse import SparseTensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import (
    IterativeRecommender,
)
from warprec.recommenders.losses import EmbLoss
from warprec.recommenders.general_recommender.graph_based import GraphRecommenderUtils
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class GCMCEncoderLayer(Module):
    """A single GCMC layer for message passing across multiple edge types (ratings).

    This layer applies a rating-specific linear transformation to the aggregated
    neighbor messages.

    Args:
        in_features (int): The dimensionality of input node features.
        out_features (int): The dimensionality of output node features.
        n_ratings (int): The number of distinct rating values (edge types).
    """

    def __init__(self, in_features: int, out_features: int, n_ratings: int):
        super().__init__()
        # A separate weight matrix for each rating type
        self.weights = nn.Parameter(torch.Tensor(n_ratings, in_features, out_features))
        nn.init.xavier_uniform_(self.weights)

    def forward(
        self,
        user_features: Tensor,
        item_features: Tensor,
        adj_tensors: List[SparseTensor],
    ) -> Tuple[Tensor, Tensor]:
        """Performs the message passing for one layer."""
        user_outputs, item_outputs = [], []

        # Iterate over each rating type (edge type)
        for r, adj_r in enumerate(adj_tensors):
            weight_r = self.weights[r]  # [in_features, out_features]

            # Message passing: Items -> Users
            # adj_r is [num_users, num_items], item_features is [num_items, in_features]
            # user_msg becomes [num_users, in_features]
            user_msg = adj_r.matmul(item_features, reduce="sum")
            user_outputs.append(torch.matmul(user_msg, weight_r))

            # Message passing: Users -> Items
            # adj_r.t() is [num_items, num_users], user_features is [num_users, in_features]
            # item_msg becomes [num_items, in_features]
            item_msg = adj_r.t().matmul(user_features, reduce="sum")
            item_outputs.append(torch.matmul(item_msg, weight_r))

        # Sum the outputs from all rating types (Accumulation)
        # Stacks to [num_ratings, num_nodes, out_features], then sums over dim 0
        final_user_output = torch.stack(user_outputs).sum(dim=0)
        final_item_output = torch.stack(item_outputs).sum(dim=0)

        return final_user_output, final_item_output


@model_registry.register(name="GCMC")
class GCMC(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of GCMC algorithm from
        Graph Convolutional Matrix Completion (KDD 2018).

    For further details, check the `paper <https://arxiv.org/abs/1706.02263>`_.

    This model is a graph autoencoder for explicit feedback. It uses a graph
    convolutional encoder to learn user/item embeddings and a decoder to
    predict rating probabilities.

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
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition for explicit feedback
    DATALOADER_TYPE = DataLoaderType.ITEM_RATING_LOADER

    # Model hyperparameters
    embedding_size: int
    reg_weight: float
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
            raise ValueError("Users value must be provided.")
        self.n_items = info.get("items", None)
        if not self.n_items:
            raise ValueError("Items value must be provided.")

        self.block_size = kwargs.get("block_size", 50)

        # Determine the unique ratings in the dataset
        unique_ratings = interactions.get_unique_ratings()
        self.n_ratings = len(unique_ratings)
        if self.n_ratings == 0:
            raise ValueError(
                "GCMC model requires explicit feedback with at least one rating value."
            )

        # Map rating values to class indices for loss computation
        classes_tensor = torch.tensor(unique_ratings, dtype=torch.float32)
        self.register_buffer("classes_tensor", classes_tensor)

        # Initial node features (embeddings)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # Encoder and decoder modules
        self.encoder = GCMCEncoderLayer(
            self.embedding_size, self.embedding_size, self.n_ratings
        )
        self.decoder = nn.Linear(2 * self.embedding_size, self.n_ratings)

        # Create adjacency matrices for each rating
        self.adj_tensors = []
        for rating_value in unique_ratings:
            # Retrieve the adjacency matrix for this rating
            matrix = interactions.get_sparse_by_rating(rating_value).tocoo()

            # Extract row and column indices
            row = torch.from_numpy(matrix.row).long()
            col = torch.from_numpy(matrix.col).long()

            # Create rectangular SparseTensor (Bipartite graph)
            adj_tensor = SparseTensor(
                row=row, col=col, sparse_sizes=(self.n_users, self.n_items + 1)
            )

            self.adj_tensors.append(adj_tensor)

        self.apply(self._init_weights)
        self.ce_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return interactions.get_item_rating_dataloader(
            neg_samples=0, batch_size=self.batch_size, low_memory=low_memory
        )

    def propagate_embeddings(self) -> Tuple[Tensor, Tensor]:
        """Performs the graph convolution to get final node embeddings."""
        user_feat = self.user_embedding.weight
        item_feat = self.item_embedding.weight

        # Move adjacency tensors to the same device as features
        device = user_feat.device
        if self.adj_tensors[0].device() != device:
            self.adj_tensors = [adj.to(device) for adj in self.adj_tensors]

        user_embed, item_embed = self.encoder(user_feat, item_feat, self.adj_tensors)

        return F.relu(user_embed), F.relu(item_embed)

    def train_step(self, batch: Any, *args: Any, **kwargs: Any):
        user, item, rating = batch

        predictions = self.forward(user, item)

        # Find the closest class index for each true rating in a vectorized way
        diff = torch.abs(
            rating.unsqueeze(1) - self.classes_tensor.unsqueeze(0)  # type: ignore[operator]
        )  # [batch_size, num_ratings]

        # Calculate CE loss
        _, target_classes = torch.min(diff, dim=1)  # [batch_size]
        ce_loss = self.ce_loss(predictions, target_classes)

        # Calculate L2 regularization
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(item),
        )

        return ce_loss + reg_loss

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Forward pass for GCMC. Computes rating logits for given user-item pairs.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.

        Returns:
            Tensor: The raw logits for each rating class for each pair.
        """
        # Get the final, propagated embeddings for all users and items
        user_all_embeddings, item_all_embeddings = self.propagate_embeddings()

        # Select embeddings for the current batch
        user_e = user_all_embeddings[user]
        item_e = item_all_embeddings[item]

        # Concatenate user and item embeddings
        combined_e = torch.cat([user_e, item_e], dim=1)

        # Pass through the decoder to get rating logits
        logits = self.decoder(combined_e)
        return logits

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the learned embeddings.

        Returns the *expected rating* for each user-item pair, used for ranking.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item} containing expected ratings.
        """
        # Perform graph propagation once to get all node embeddings
        user_all_embeddings, item_all_embeddings = self.propagate_embeddings()

        # Deconstruct the decoder's linear layer for efficient calculation
        w = self.decoder.weight  # [num_ratings, 2*emb]
        b = self.decoder.bias  # [num_ratings]

        # Split weights into user and item parts
        w_user, w_item = torch.split(
            w, self.embedding_size, dim=1
        )  # [num_ratings, embedding_size]

        # Pre-compute partial scores for users in this batch
        batch_user_part = F.linear(
            user_all_embeddings[user_indices], w_user
        )  # [batch_size, num_ratings]

        if item_indices is None:
            # Case 'full': iterate through all items in memory-safe blocks
            valid_items_emb = item_all_embeddings[:-1]

            # Pre-compute the item-dependent part for ALL valid items
            all_items_part = F.linear(
                valid_items_emb, w_item
            )  # [num_items, num_ratings]

            all_scores = []
            n_items = valid_items_emb.size(0)

            # Add bias to user part once to avoid adding it in the loop
            batch_user_part_with_bias = batch_user_part + b

            for start in range(0, n_items, self.block_size):
                end = min(start + self.block_size, n_items)

                # Slice pre-computed item parts
                item_part_block = all_items_part[start:end]  # [block, num_ratings]

                # Use broadcasting to efficiently compute logits for the block
                logits_block = batch_user_part_with_bias.unsqueeze(
                    1
                ) + item_part_block.unsqueeze(0)  # [batch_size, block, num_ratings]

                probs_block = F.softmax(logits_block, dim=2)

                # Calculate expected rating: sum(probs * rating_values)
                expected_ratings = torch.einsum(
                    "bif,f->bi", probs_block, self.classes_tensor
                )  # [batch_size, block]
                all_scores.append(expected_ratings)

            predictions = torch.cat(all_scores, dim=1)  # [batch_size, num_items]
            return predictions

        # Case 'sampled': process all given item_indices at once
        batch_item_emb = item_all_embeddings[
            item_indices
        ]  # [batch_size, pad_seq, embedding_size]

        # Compute item part for sampled items
        # [B, S, E] @ [E, R] -> [B, S, R]
        # B=batch_size, S=pad_seq, E=embedding_size, R=num_ratings
        batch_item_part = torch.matmul(batch_item_emb, w_item.t())

        # Sum parts and bias to get logits
        # [B, 1, R] + [B, S, R] + [R] -> [B, S, R]
        logits = batch_user_part.unsqueeze(1) + batch_item_part + b

        probs = F.softmax(logits, dim=2)

        # Calculate expected rating
        # [B, S, R] * [R] -> [B, S]
        predictions = torch.einsum("bif,f->bi", probs, self.classes_tensor)
        return predictions
