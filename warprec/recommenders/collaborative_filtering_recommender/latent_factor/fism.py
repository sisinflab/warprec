# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
from torch import nn, Tensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="FISM")
class FISM(IterativeRecommender):
    r"""Implementation of FISM model from
    FISM: Factored Item Similarity Models for Top-N Recommender Systems (KDD 2013).

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/2487575.2487589>`_.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The number of factors for item feature embeddings.
        alpha (float): The alpha parameter, a value between 0 and 1,
            used in the similarity calculation.
        split_to (int): Parameter for splitting items into chunks
            during prediction (for memory management).
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The size of the batches used during training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.HISTORY

    # Model specific parameters
    embedding_size: int
    alpha: float
    split_to: int
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

        # Embeddings and biases
        self.item_src_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )
        self.item_dst_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )
        self.user_bias = nn.Parameter(torch.zeros(self.n_users))
        self.item_bias = nn.Parameter(torch.zeros(self.n_items + 1))  # +1 for padding

        # Prepare history information
        history_matrix, history_lens, history_mask = interactions.get_history()

        # Use buffers to store non-trainable tensors
        self.register_buffer("history_matrix", history_matrix)
        self.register_buffer("history_lens", history_lens)
        self.register_buffer("history_mask", history_mask)

        # Handle groups
        self.group = torch.chunk(torch.arange(1, self.n_items + 1), self.split_to)

        # Init embedding weights
        self.apply(self._init_weights)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = EmbLoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return interactions.get_item_rating_dataloader(
            neg_samples=0,
            batch_size=self.batch_size,
            low_memory=low_memory,
        )

    def train_step(self, batch: Any, *args, **kwargs):
        user, item, rating = batch

        # Calculate BCE loss
        predictions = self(user, item)
        bce_loss = self.bce_loss(predictions, rating)

        # Calculate L2 regularization
        reg_loss = self.reg_weight * self.reg_loss(
            self.item_src_embedding(item),
            self.item_dst_embedding(item),
        )

        return bce_loss + reg_loss

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Forward pass for calculating scores for specific user-item pairs.

        Args:
            user (Tensor): User indices.
            item (Tensor): Item indices.

        Returns:
            Tensor: Predicted scores.
        """
        user_inter = self.history_matrix[user]  # type: ignore[index]
        item_num = self.history_lens[user].unsqueeze(1)  # type: ignore[index]
        batch_mask_mat = self.history_mask[user]  # type: ignore[index]

        user_history = self.item_src_embedding(
            user_inter
        )  # batch_size x max_len x embedding_size
        target = self.item_dst_embedding(item)  # batch_size x embedding_size

        user_bias = self.user_bias[user]  # batch_size
        item_bias = self.item_bias[item]  # batch_size

        # (batch_size, max_len, embedding_size) @ (batch_size, embedding_size, 1) -> (batch_size, max_len, 1)
        similarity = torch.bmm(user_history, target.unsqueeze(2)).squeeze(
            2
        )  # batch_size x max_len

        # Apply mask to similarity
        similarity = batch_mask_mat * similarity

        # coeff = N_u ^ (-alpha)
        # Add a small epsilon to item_num to prevent division by zero for users with no history
        coeff = torch.pow(item_num.squeeze(1).float() + 1e-6, -self.alpha)  # batch_size

        # Scores = coeff * sum(similarity) + user_bias + item_bias
        scores = coeff * torch.sum(similarity, dim=1) + user_bias + item_bias
        return scores

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
        # Select data for current batch
        batch_history_matrix = self.history_matrix[user_indices]  # type: ignore[index]
        batch_history_lens = self.history_lens[user_indices]  # type: ignore[index]
        batch_history_mask = self.history_mask[user_indices]  # type: ignore[index]
        batch_user_bias = self.user_bias[user_indices]

        # Compute aggregated embedding for user in batch
        user_history_emb = self.item_src_embedding(
            batch_history_matrix
        )  # [batch_size, max_len, embedding_size]

        # Apply masking
        masked_user_history_emb = (
            user_history_emb * batch_history_mask.unsqueeze(2).float()
        )
        user_aggregated_emb = masked_user_history_emb.sum(
            dim=1
        )  # [batch_size, embedding_size]

        # Normalization coefficient (N_u ^ -alpha)
        coeff = torch.pow(batch_history_lens.float() + 1e-6, -self.alpha).unsqueeze(1)
        user_final_emb = user_aggregated_emb * coeff  # [batch_size, embedding_size]

        if item_indices is None:
            # Case 'full': prediction on all items
            item_dst_embeddings = self.item_dst_embedding.weight[
                :-1, :
            ]  # [n_items, embedding_size]
            item_biases = self.item_bias[:-1]  # [n_items]
            einsum_string = "be,ie->bi"  # b: batch, e: embedding, i: item
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_dst_embeddings = self.item_dst_embedding(
                item_indices
            )  # [batch_size, pad_seq, embedding_size]
            item_biases = self.item_bias[item_indices]  # [batch_size, pad_seq]
            einsum_string = "be,bse->bs"  # b: batch, e: embedding, s: sample

        # Compute prediction step
        predictions = torch.einsum(
            einsum_string, user_final_emb, item_dst_embeddings
        )  # [batch_size, n_items] or [batch_size, pad_seq]

        # Add the biases
        predictions += batch_user_bias.unsqueeze(1)
        predictions += item_biases
        return predictions
