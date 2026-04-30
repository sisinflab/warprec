# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
from typing import Any, Optional

import torch
from torch import nn, Tensor

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import EmbLoss
from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class BSARecEncoder(nn.Module):
    """BSARec Encoder: Stacked layers combining frequency filtering and attention."""

    def __init__(
        self,
        embedding_size: int,
        n_layers: int,
        n_heads: int,
        inner_size: int,
        attn_dropout_prob: float,
        dropout_prob: float,
        max_seq_len: int,
        alpha: float = 0.5,
        c: int = 10,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                BSARecBlock(
                    embedding_size=embedding_size,
                    n_heads=n_heads,
                    inner_size=inner_size,
                    attn_dropout_prob=attn_dropout_prob,
                    dropout_prob=dropout_prob,
                    max_seq_len=max_seq_len,
                    alpha=alpha,
                    c=c,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self, hidden_states: Tensor, padding_mask: Tensor, attn_mask: Tensor
    ) -> Tensor:
        """Pass through stacked BSARec blocks."""
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, padding_mask, attn_mask)
        return hidden_states


class BSARecBlock(nn.Module):
    """Single BSARec Block: Frequency Layer + Attention Layer + FeedForward."""

    def __init__(
        self,
        embedding_size: int,
        n_heads: int,
        inner_size: int,
        attn_dropout_prob: float,
        dropout_prob: float,
        max_seq_len: int,
        alpha: float = 0.5,
        c: int = 10,
    ):
        super().__init__()
        self.frequency_layer = FrequencyLayer(embedding_size, dropout_prob, c)
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=n_heads,
            dropout=attn_dropout_prob,
            batch_first=True,
        )
        self.feed_forward = FeedForwardLayer(embedding_size, inner_size, dropout_prob)
        self.layernorm1 = nn.LayerNorm(embedding_size, eps=1e-8)
        self.layernorm2 = nn.LayerNorm(embedding_size, eps=1e-8)
        self.dropout = nn.Dropout(dropout_prob)
        self.alpha = alpha  # Balance parameter: alpha * AIB + (1-alpha) * attention

    def forward(
        self, hidden_states: Tensor, padding_mask: Tensor, attn_mask: Tensor
    ) -> Tensor:
        """
        Forward pass of a BSA block.

        Args:
            hidden_states (Tensor): [batch_size, seq_len, embedding_size]
            padding_mask (Tensor): [batch_size, seq_len] - True for padding tokens
            attn_mask (Tensor): [seq_len, seq_len] - True where future positions are masked

        Returns:
            Tensor: [batch_size, seq_len, embedding_size]
        """
        # The paper defines the attentive inductive bias and self-attention
        # as parallel branches applied to the same layer input X^l.
        aib_output = self.frequency_layer(hidden_states)
        attn_output, _ = self.attention_layer(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        # Eq. (5): alpha * A_IB X^l + (1 - alpha) * A X^l
        combined = self.alpha * aib_output + (1 - self.alpha) * attn_output
        combined = self.layernorm1(hidden_states + self.dropout(combined))

        ff_output = self.feed_forward(combined)
        output = self.layernorm2(combined + self.dropout(ff_output))
        return output


class FrequencyLayer(nn.Module):
    """Frequency Layer: FFT-based low-pass + high-pass decomposition.

    Captures periodic and cyclical patterns in sequences using spectral analysis.
    The learnable sqrt_beta parameter allows the model to adjust the contribution
    of high-frequency components.
    """

    def __init__(self, embedding_size: int, dropout_prob: float, c: int = 10):
        super().__init__()
        self.c = c  # Cutoff frequency for low-pass filter
        self.dropout = nn.Dropout(dropout_prob)
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, embedding_size))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Apply FFT-based frequency filtering.

        Args:
            hidden_states (Tensor): [batch_size, seq_len, embedding_size]

        Returns:
            Tensor: Frequency-filtered output [batch_size, seq_len, embedding_size]
        """
        _, seq_len, _ = hidden_states.shape

        # Compute FFT along sequence dimension
        # rfft: real FFT, returns complex values
        x_fft = torch.fft.rfft(hidden_states, dim=1, norm="ortho")

        # Low-pass filter: keep only low frequencies
        low_pass = x_fft.clone()
        low_pass[:, self.c :, :] = 0  # Zero out frequencies > c
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm="ortho")

        # High-pass: residual from low-pass
        high_pass = hidden_states - low_pass

        # Combine with learnable scaling
        # sqrt_beta^2 controls contribution of high-frequency components
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        output = self.dropout(sequence_emb_fft)
        return output


class FeedForwardLayer(nn.Module):
    """Feed-Forward Network with two linear layers and GELU activation."""

    def __init__(self, embedding_size: int, inner_size: int, dropout_prob: float):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, inner_size)
        self.linear2 = nn.Linear(inner_size, embedding_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply feed-forward transformation."""
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


@model_registry.register(name="BSARec")
class BSARec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of BASRec model from
    "BSARec: Bandlimited Self-Attention for Sequential Recommendation." in AAAi 2024.

    This model combines frequency-based filtering with self-attention to capture
    both periodic patterns and sequential dependencies in user behavior.

    Architecture:
    1. Domain-Specific Patterns (DSP): FFT-based low/high-pass filtering
    2. Graph-Space Patterns (GSP): Multi-head self-attention
    3. Adaptive Combination: Learnable weighted sum (alpha parameter)

    The frequency filtering helps capture cyclical patterns (e.g., weekly habits),
    while attention captures complex sequential dependencies.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The dimension of the item embeddings (hidden_size).
        n_layers (int): The number of transformer encoder layers.
        n_heads (int): The number of attention heads in the transformer.
        inner_size (int): The dimensionality of the feed-forward layer.
        dropout_prob (float): The probability of dropout for embeddings.
        attn_dropout_prob (float): The probability of dropout for attention weights.
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The value of weight decay used in optimizer.
        batch_size (int): The batch size used during training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
        alpha (float): Balance parameter between DSP and GSP (0.0-1.0).
        c (int): Cutoff frequency for low-pass filtering.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.SEQUENTIAL_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    n_heads: int
    inner_size: int
    dropout_prob: float
    attn_dropout_prob: float
    reg_weight: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float
    neg_samples: int
    max_seq_len: int
    alpha: float
    c: int

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        # Item and position embeddings
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)

        # BSARec layers: frequency filtering + attention
        self.bsarec_encoder = BSARecEncoder(
            embedding_size=self.embedding_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            inner_size=self.inner_size,
            attn_dropout_prob=self.attn_dropout_prob,
            dropout_prob=self.dropout_prob,
            max_seq_len=self.max_seq_len,
            alpha=self.alpha,
            c=self.c,
        )

        # Precompute causal mask
        causal_mask = self._generate_square_subsequent_mask(self.max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

        # The paper optimizes next-item prediction with full softmax CE loss.
        self.main_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return sessions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            neg_samples=0,
            batch_size=self.batch_size,
            low_memory=low_memory,
        )

    def training_step(self, batch: Any, batch_idx: int):
        if len(batch) == 4:
            item_seq, item_seq_len, pos_item, neg_item = batch
        else:
            item_seq, item_seq_len, pos_item = batch
            neg_item = None

        seq_output = self.forward(item_seq, item_seq_len)

        logits = torch.matmul(
            seq_output, self.item_embedding.weight[:-1].transpose(0, 1)
        )
        main_loss = self.main_loss(logits, pos_item)

        reg_terms = [
            self.item_embedding(item_seq),
            self.item_embedding(pos_item),
        ]
        if neg_item is not None:
            reg_terms.append(self.item_embedding(neg_item))
        reg_loss = self.reg_weight * self.reg_loss(*reg_terms)

        return main_loss + reg_loss

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        """Forward pass of the BSARec model.

        Combines frequency-based patterns (DSP) with attention patterns (GSP)
        through an adaptive weighted combination.

        Args:
            item_seq (Tensor): Padded sequences of item IDs [batch_size, max_seq_len].
            item_seq_len (Tensor): Actual lengths of sequences [batch_size,].

        Returns:
            Tensor: The embedding of the predicted item (last session state)
                    [batch_size, embedding_size].
        """
        seq_len = item_seq.size(1)

        # Padding mask to ignore padding tokens
        padding_mask = item_seq == self.n_items

        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long).to(item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        # Get embeddings
        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(position_ids)

        # Combine embeddings and apply LayerNorm + Dropout
        seq_emb = self.layernorm(item_emb + pos_emb)
        seq_emb = self.emb_dropout(seq_emb)

        # Pass through BSARec encoder (frequency + attention layers)
        transformer_output = self.bsarec_encoder(
            seq_emb,
            padding_mask,
            self.causal_mask[:seq_len, :seq_len],  # type: ignore[index]
        )  # [batch_size, max_seq_len, embedding_size]

        # Gather the output of the last relevant item in each sequence
        seq_output = self._gather_indexes(
            transformer_output, item_seq_len - 1
        )  # [batch_size, embedding_size]

        return seq_output

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        user_seq: Optional[Tensor] = None,
        seq_len: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Prediction using the learned session embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            user_seq (Optional[Tensor]): Padded sequences of item IDs for users to predict for.
            seq_len (Optional[Tensor]): Actual lengths of these sequences, before padding.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Get sequence output embeddings
        seq_output = self.forward(user_seq, seq_len)

        if item_indices is None:
            # Case 'full': prediction on all items
            item_embeddings = self.item_embedding.weight[:-1, :]
            einsum_string = "be,ie->bi"
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = self.item_embedding(item_indices)
            einsum_string = "be,bse->bs"

        predictions = torch.einsum(einsum_string, seq_output, item_embeddings)
        return predictions
