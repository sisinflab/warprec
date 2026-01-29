# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class LightSANsLayer(nn.Module):
    """Implements the Low-Rank Decomposed Self-Attention and Decoupled Position Encoding."""

    def __init__(
        self,
        n_heads: int,
        k_interests: int,
        hidden_size: int,
        inner_size: int,
        dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.k_interests = k_interests
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads

        # Item-to-Interest Parameters (Theta)
        self.w_theta = nn.Parameter(torch.Tensor(hidden_size, k_interests))

        # Semantic Projections (Content)
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # Position Projections (Decoupled)
        self.w_pos_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_pos_k = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output & FFN
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(inner_size, hidden_size),
            nn.Dropout(dropout_prob),
        )

        self.dropout = nn.Dropout(dropout_prob)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.w_theta)

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        padding_mask: Optional[Tensor] = None,
        causal_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of LightSANs Layer.

        Args:
            x (Tensor): Item embeddings [batch_size, seq_len, hidden_size]
            pos_emb (Tensor): Position embeddings [seq_len, hidden_size] (Shared across batch)
            padding_mask (Optional[Tensor]): [batch_size, seq_len] (True where padding)
            causal_mask (Optional[Tensor]): [seq_len, seq_len]

        Returns:
            Tensor: Output of the forward pass.
        """
        B, L, _ = x.shape

        # Projections
        # [batch_size, n_heads, seq_len, head_dim]
        q = self.w_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Item-to-Interest Aggregation
        # Calculate relevance D: [batch_size, seq_len, k]
        d_logits = torch.matmul(x, self.w_theta)

        # Apply padding mask to D to avoid aggregating padding items into interests
        if padding_mask is not None:
            # padding_mask is [batch_size, seq_len], reshape to [batch_size, seq_len, 1]
            d_logits = d_logits.masked_fill(padding_mask.unsqueeze(-1), -1e9)

        d_scores = F.softmax(d_logits, dim=1)  # Softmax over seq_len

        # Aggregate K and V into Interests: [batch_size, n_heads, k, head_dim]
        # Transpose D for aggregation: [batch_size, 1, k, seq_len]
        d_scores_t = d_scores.transpose(1, 2).unsqueeze(1)

        k_tilde = torch.matmul(d_scores_t, k)
        v_tilde = torch.matmul(d_scores_t, v)

        # Item-to-Interest Interaction
        attn_item_scores = torch.matmul(q, k_tilde.transpose(-1, -2))
        attn_item_scores = attn_item_scores / (self.head_dim**0.5)
        attn_item_probs = F.softmax(attn_item_scores, dim=-1)
        attn_item_probs = self.attn_dropout(attn_item_probs)

        # Context Representation: A_item * V_tilde
        context_layer = torch.matmul(attn_item_probs, v_tilde)

        # Decoupled Position Encoding
        # Project positions [seq_len, n_heads, head_dim]
        # pos_emb is [seq_len, n_heads]
        pos_q = (
            self.w_pos_q(pos_emb).view(L, self.n_heads, self.head_dim).transpose(0, 1)
        )  # [n_heads, seq_len, head_dim]
        pos_k = (
            self.w_pos_k(pos_emb).view(L, self.n_heads, self.head_dim).transpose(0, 1)
        )  # [n_heads, seq_len, head_dim]

        # A_pos = softmax(Pos_Q * Pos_K^T / sqrt(d))
        # [n_heads, seq_len, head_dim] * [n_heads, head_dim, seq_len] -> [n_heads, seq_len, seq_len]
        attn_pos_scores = torch.matmul(pos_q, pos_k.transpose(-1, -2))
        attn_pos_scores = attn_pos_scores / (self.head_dim**0.5)

        # Apply Causal Mask to Position Attention
        if causal_mask is not None:
            attn_pos_scores = attn_pos_scores.masked_fill(causal_mask == 0, -1e9)

        attn_pos_probs = F.softmax(attn_pos_scores, dim=-1)
        attn_pos_probs = self.attn_dropout(attn_pos_probs)

        # Position Context: A_pos * V (Original V)
        # We unsqueeze dim 0 to broadcast across batch
        pos_context = torch.matmul(attn_pos_probs.unsqueeze(0), v)

        # Combine and Output
        output = context_layer + pos_context

        # Reshape and Project
        output = output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        output = self.out_proj(output)
        output = self.dropout(output)

        # Residual + Norm
        x = self.layer_norm1(x + output)

        # FFN
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)

        return x


@model_registry.register(name="LightSANs")
class LightSANs(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of LightSANs algorithm from
    "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation" (SIGIR 2021).

    For further details, check the `paper <https://dl.acm.org/doi/10.1145/3404835.3462978>`_.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The dimension of the item embeddings.
        n_layers (int): The number of attention layers.
        n_heads (int): The number of attention heads.
        k_interests (int): The number of latent interests (k).
        inner_size (int): The dimensionality of the feed-forward layer.
        dropout_prob (float): The probability of dropout.
        attn_dropout_prob (float): The probability of dropout for attention.
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used during training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.SEQUENTIAL_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    n_heads: int
    k_interests: int
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

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)

        # LightSANs Layers
        self.layers = nn.ModuleList(
            [
                LightSANsLayer(
                    n_heads=self.n_heads,
                    k_interests=self.k_interests,
                    hidden_size=self.embedding_size,
                    inner_size=self.inner_size,
                    dropout_prob=self.dropout_prob,
                    attn_dropout_prob=self.attn_dropout_prob,
                    layer_norm_eps=1e-8,
                )
                for _ in range(self.n_layers)
            ]
        )

        # Precompute causal mask
        causal_mask = self._generate_square_subsequent_mask(self.max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

        # Initialize weights
        self.apply(self._init_weights)

        # Loss function
        self.main_loss: nn.Module
        if self.neg_samples > 0:
            self.main_loss = BPRLoss()
        else:
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
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
            low_memory=low_memory,
        )

    def train_step(self, batch: Any, *args, **kwargs):
        if self.neg_samples > 0:
            item_seq, item_seq_len, pos_item, neg_item = batch
        else:
            item_seq, item_seq_len, pos_item = batch
            neg_item = None

        seq_output = self.forward(item_seq, item_seq_len)

        # Calculate main loss and L2 regularization
        if self.neg_samples > 0:
            pos_items_emb = self.item_embedding(pos_item)
            neg_items_emb = self.item_embedding(neg_item)

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output.unsqueeze(1) * neg_items_emb, dim=-1)
            main_loss = self.main_loss(pos_score, neg_score)

            # L2 regularization
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq),
                self.item_embedding(pos_item),
                self.item_embedding(neg_item),
            )
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            main_loss = self.main_loss(logits, pos_item)

            # L2 regularization
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq),
                self.item_embedding(pos_item),
            )

        return main_loss + reg_loss

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        """Forward pass of the LightSANs model.

        Args:
            item_seq (Tensor): Padded sequences of item IDs [batch_size, max_seq_len].
            item_seq_len (Tensor): Actual lengths of sequences [batch_size,].

        Returns:
            Tensor: The embedding of the predicted item (last session state).
        """
        seq_len = item_seq.size(1)

        # Padding mask (True where padding exists)
        padding_mask = item_seq == self.n_items

        # Get embeddings
        x = self.item_embedding(item_seq)
        x = self.layernorm(x)
        x = self.emb_dropout(x)

        # Get Position Embeddings for the current sequence length
        # These are shared across the batch
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        pos_emb = self.position_embedding(position_ids)

        # Causal mask for the current sequence length
        # We slice the precomputed mask
        curr_causal_mask = self.causal_mask[:seq_len, :seq_len]  # type: ignore[index]

        # Pass through LightSANs Layers
        for layer in self.layers:
            x = layer(
                x,
                pos_emb=pos_emb,
                padding_mask=padding_mask,
                causal_mask=curr_causal_mask,
            )

        # Gather the output of the last relevant item in each sequence
        seq_output = self._gather_indexes(x, item_seq_len - 1)

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
        seq_output = self.forward(user_seq, seq_len)  # [batch_size, embedding_size]

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

        predictions = torch.einsum(
            einsum_string, seq_output, item_embeddings
        )  # [batch_size, n_items] or [batch_size, pad_seq]
        return predictions
