# pylint: disable = R0801, E1102
from typing import Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import normal_

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="FISM")
class FISM(IterativeRecommender):
    r"""Implementation of FISM model from
    FISM: Factored Item Similarity Models for Top-N Recommender Systems (KDD 2013).

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/2487575.2487589>`_.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items or users value was not passed through the info dict.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The number of factors for item feature embeddings.
        alpha (float): The alpha parameter, a value between 0 and 1, used in the similarity calculation.
        split_to (int): Parameter for splitting items into chunks during prediction (for memory management).
        weight_decay (float): The value of weight decay used in the optimizer.
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
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(
            params, interactions, *args, device=device, seed=seed, info=info, **kwargs
        )

        # Get information from dataset info
        users = info.get("users", None)
        if not users:
            raise ValueError(
                "Users value must be provided to correctly initialize the model."
            )
        items = info.get("items", None)
        if not items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )

        # Embeddings and biases
        self.item_src_embedding = nn.Embedding(
            items + 1, self.embedding_size, padding_idx=items
        )
        self.item_dst_embedding = nn.Embedding(
            items + 1, self.embedding_size, padding_idx=items
        )
        self.user_bias = nn.Parameter(torch.zeros(users))
        self.item_bias = nn.Parameter(torch.zeros(items + 1))  # +1 for padding

        # Prepare history information
        self.history_matrix, self.history_lens, self.history_mask = (
            interactions.get_history()
        )

        # Handle groups
        self.group = torch.chunk(torch.arange(1, items + 1), self.split_to)

        # Init embedding weights
        self.apply(self._init_weights)
        self.loss = nn.BCEWithLogitsLoss()

        # Move to device
        self.to(self._device)

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, 0, 0.01)  # Standard deviation

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
        user, item, rating = [x.to(self._device) for x in batch]

        predictions = self(user, item)
        loss: Tensor = self.loss(predictions, rating)

        return loss

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Forward pass for calculating scores for specific user-item pairs.

        Args:
            user (Tensor): User indices.
            item (Tensor): Item indices.

        Returns:
            Tensor: Predicted scores.
        """
        user_inter = self.history_matrix[user]
        item_num = self.history_lens[user].unsqueeze(1)
        batch_mask_mat = self.history_mask[user]

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
    def predict_full(
        self,
        user_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the learned embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Retrieve embeddings + biases
        all_item_src_emb = self.item_src_embedding.weight
        all_item_dst_emb = self.item_dst_embedding.weight[:-1, :]
        all_user_bias = self.user_bias
        all_item_bias = self.item_bias[:-1]

        # Select data for current batch
        batch_history_matrix = self.history_matrix[user_indices]
        batch_history_lens = self.history_lens[user_indices]
        batch_history_mask = self.history_mask[user_indices]
        batch_user_bias = all_user_bias[user_indices]

        # Compute aggregated embedding for user in batch
        user_history_emb = all_item_src_emb[
            batch_history_matrix
        ]  # [batch_size, max_len, emb_size]

        # Apply masking
        masked_user_history_emb = (
            user_history_emb * batch_history_mask.unsqueeze(2).float()
        )
        user_aggregated_emb = masked_user_history_emb.sum(
            dim=1
        )  # [batch_size, emb_size]

        # Normalization coefficient (N_u ^ -alpha)
        coeff = torch.pow(batch_history_lens.float() + 1e-6, -self.alpha).unsqueeze(1)
        user_final_emb = user_aggregated_emb * coeff  # [batch_size, emb_size]

        # Compute the final matrix multiplication.
        predictions = torch.matmul(
            user_final_emb, all_item_dst_emb.transpose(0, 1)
        )  # [batch_size, n_items]

        # Add the bias
        predictions += batch_user_bias.unsqueeze(1)
        predictions += all_item_bias.unsqueeze(0)
        return predictions.to(self._device)

    @torch.no_grad()
    def predict_sampled(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction of given items using the learned embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            item_indices (Tensor): The batch of item indices.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x pad_seq}.
        """
        # Select data for current batch
        batch_history_matrix = self.history_matrix[user_indices]
        batch_history_lens = self.history_lens[user_indices]
        batch_history_mask = self.history_mask[user_indices]
        batch_user_bias = self.user_bias[user_indices]

        # Compute aggregated embedding for user in batch
        user_history_emb = self.item_src_embedding(
            batch_history_matrix
        )  # [batch_size, max_len, emb_size]

        # Apply masking
        masked_user_history_emb = (
            user_history_emb * batch_history_mask.unsqueeze(2).float()
        )
        user_aggregated_emb = masked_user_history_emb.sum(
            dim=1
        )  # [batch_size, emb_size]

        # Normalization coefficient (N_u ^ -alpha)
        coeff = torch.pow(batch_history_lens.float() + 1e-6, -self.alpha).unsqueeze(1)
        user_final_emb = user_aggregated_emb * coeff  # [batch_size, emb_size]

        # Retrieve embeddings for candidate items, handling padding (-1)
        # We need to add 1 to the item indices because 0 is the padding index
        candidate_item_embeddings = self.item_dst_embedding(
            item_indices
        )  # [batch_size, pad_seq, embedding_size]
        candidate_item_biases = self.item_bias[item_indices]  # [batch_size, pad_seq]

        # Compute the final matrix multiplication using einsum
        predictions = torch.einsum(
            "bi,bji->bj", user_final_emb, candidate_item_embeddings
        )

        # Add the biases
        predictions += batch_user_bias.unsqueeze(1)
        predictions += candidate_item_biases
        return predictions.to(self._device)
