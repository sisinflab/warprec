# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
from typing import Any

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_normal_

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss
from warprec.data.dataset import Interactions, Sessions
from warprec.utils.registry import model_registry


@model_registry.register(name="SASRec")
class SASRec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of SASRec algorithm from
    "Self-Attentive Sequential Recommendation." in ICDM 2018.

    This implementation is adapted to the WarpRec framework, using PyTorch's
    native nn.TransformerEncoder for the self-attention mechanism.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items value was not passed through the info dict.

    Attributes:
        embedding_size (int): The dimension of the item embeddings (hidden_size).
        n_layers (int): The number of transformer encoder layers.
        n_heads (int): The number of attention heads in the transformer.
        inner_size (int): The dimensionality of the feed-forward layer in the transformer.
        dropout_prob (float): The probability of dropout for embeddings and other layers.
        attn_dropout_prob (float): The probability of dropout for the attention weights.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used during training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
    """

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    n_heads: int
    inner_size: int
    dropout_prob: float
    attn_dropout_prob: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float
    neg_samples: int
    max_seq_len: int

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, *args, **kwargs)

        items = info.get("items", None)
        if not items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )

        self.item_embedding = nn.Embedding(
            items + 1, self.embedding_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.attn_dropout_prob,
            activation="gelu",  # GELU is a common choice in Transformers
            batch_first=True,  # Input tensors are (batch, seq_len, features)
            norm_first=False,  # Following the original Transformer paper (post-LN)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Loss function will be based on number of
        # negative samples
        self.loss: nn.Module
        if self.neg_samples > 0:
            self.loss = BPRLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self.to(self._device)

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, (nn.Embedding, nn.Linear)):
            # Using Xavier Normal initialization for consistency with the framework
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.LayerNorm):
            # Standard initialization for LayerNorm
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _generate_square_subsequent_mask(self, seq_len: int) -> Tensor:
        """Generates a causal mask for the transformer."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=self._device), diagonal=1)
        return mask.bool()  # True values will be masked

    def get_dataloader(self, interactions: Interactions, sessions: Sessions, **kwargs):
        return sessions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            num_negatives=self.neg_samples,
            batch_size=self.batch_size,
        )

    def train_step(self, batch: Any, *args, **kwargs):
        if self.neg_samples > 0:
            item_seq, item_seq_len, pos_item, neg_item = [
                x.to(self._device) for x in batch
            ]
        else:
            item_seq, item_seq_len, pos_item = [x.to(self._device) for x in batch]
            neg_item = None

        seq_output = self.forward(item_seq, item_seq_len)

        loss: Tensor
        if self.neg_samples > 0:
            pos_items_emb = self.item_embedding(
                pos_item
            )  # [batch_size, embedding_size]
            neg_items_emb = self.item_embedding(
                neg_item
            )  # [batch_size, embedding_size]

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [batch_size]
            neg_score = torch.sum(
                seq_output.unsqueeze(1) * neg_items_emb, dim=-1
            )  # [batch_size]
            loss = self.loss(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight  # [num_items, embedding_size]
            logits = torch.matmul(
                seq_output, test_item_emb.transpose(0, 1)
            )  # [batch_size, num_items]
            loss = self.loss(logits, pos_item)

        return loss

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        """Forward pass of the SASRec model.

        Args:
            item_seq (Tensor): Padded sequences of item IDs [batch_size, max_seq_len].
            item_seq_len (Tensor): Actual lengths of sequences [batch_size,].

        Returns:
            Tensor: The embedding of the predicted item (last session state)
                    [batch_size, embedding_size].
        """
        seq_len = item_seq.size(1)

        # Padding mask to ignore padding tokens
        padding_mask = item_seq == 0  # [batch_size, seq_len]

        # Causal mask to prevent attending to future tokens
        causal_mask = self._generate_square_subsequent_mask(seq_len)

        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long, device=self._device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        # Get embeddings
        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(position_ids)

        # Combine embeddings and apply LayerNorm + Dropout
        seq_emb = self.layernorm(item_emb + pos_emb)
        seq_emb = self.emb_dropout(seq_emb)

        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(
            src=seq_emb,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )  # [batch_size, max_seq_len, embedding_size]

        # Gather the output of the last relevant item in each sequence
        seq_output = self._gather_indexes(
            transformer_output, item_seq_len - 1
        )  # [batch_size, embedding_size]

        return seq_output

    @torch.no_grad()
    def predict_full(
        self,
        user_indices: Tensor,
        user_seq: Tensor,
        seq_len: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Prediction using the learned session embeddings (full sort prediction).

        Args:
            user_indices (Tensor): The batch of user indices.
            user_seq (Tensor): Padded sequences of item IDs for users to predict for.
            seq_len (Tensor): Actual lengths of these sequences, before padding.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        user_seq = user_seq.to(self._device)
        seq_len = seq_len.to(self._device)

        # Get the session output embedding for each user
        seq_output = self.forward(user_seq, seq_len)

        # Get embeddings for all items
        all_item_embeddings = self.item_embedding.weight[1:]

        # Calculate scores for all items
        predictions = torch.matmul(seq_output, all_item_embeddings.transpose(0, 1))
        return predictions.to(self._device)

    @torch.no_grad()
    def predict_sampled(
        self,
        user_indices: Tensor,
        item_indices: Tensor,
        user_seq: Tensor,
        seq_len: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction of given items using the learned embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            item_indices (Tensor): The batch of item indices to predict for.
            user_seq (Tensor): Padded sequences of item IDs for users to predict for.
            seq_len (Tensor): Actual lengths of these sequences, before padding.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x pad_seq}.
        """
        # Move inputs to the correct device
        user_seq = user_seq.to(self._device)
        seq_len = seq_len.to(self._device)
        item_indices = item_indices.to(self._device)

        # Calculate the sequential output embedding for each user in the batch
        seq_output = self.forward(user_seq, seq_len)  # [batch_size, embedding_size]

        # Get embeddings for candidate items. We clamp the indices to avoid
        # out-of-bounds errors with the padding value (-1)
        candidate_item_embeddings = self.item_embedding(
            item_indices.clamp(min=0)
        )  # [batch_size, pad_seq, embedding_size]

        # Compute scores using a batch matrix multiplication or einsum
        predictions = torch.einsum(
            "bi,bji->bj", seq_output, candidate_item_embeddings
        )  # [batch_size, pad_seq]
        return predictions.to(self._device)
