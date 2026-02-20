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


class LinRecAttention(nn.Module):
    """
    Causal LinRec Attention Mechanism.

    Adapted from SIGIR '23 to be strictly causal for autoregressive training.

    Changes from naive paper implementation to prevent Data Leakage:
    1. Normalization is applied feature-wise (dim=-1) instead of sequence-wise.
    2. Uses Cumulative Sum (Prefix Sum) instead of global matrix multiplication
       to ensure position 'i' only attends to '0...i'.

    Complexity: O(N * D^2) - Linear with respect to sequence length.
    """

    def __init__(self, emb_size: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads
        self.head_dim = emb_size // n_heads

        self.w_q = nn.Linear(emb_size, emb_size)
        self.w_k = nn.Linear(emb_size, emb_size)
        self.w_v = nn.Linear(emb_size, emb_size)

        self.out_proj = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask=None):
        # x shape: [B, S, E]
        b, n, d = x.shape

        # Projections [B, H, S, D_h]
        q = self.w_q(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        # Activation (ELU) + L2 Normalization (as specified in the paper)
        q = F.elu(q)
        k = F.elu(k)

        # Normalizing along sequence (dim=-2) would cause future data leakage.
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Causal Linear Attention (The "Recurrent" trick via CumSum)
        # We compute the context matrix for each step 'i' as sum(K_j^T * V_j) for j <= i

        # Compute outer product K^T * V for each step: [B, H, N, D_h, D_h]
        # This represents the "memory update" added by each token
        kv_step = torch.einsum("bhnd,bhne->bhnde", k, v)

        # Cumulative Sum along sequence dimension (N)
        kv_prefix = torch.cumsum(kv_step, dim=2)

        # Compute Query * Context
        # q: [B, H, N, D_h]
        # kv_prefix: [B, H, N, D_h, D_h]
        # out: [B, H, N, D_h]
        out = torch.einsum("bhnd,bhnde->bhne", q, kv_prefix)

        # Final Projection
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return self.out_proj(out)


class LinRecBlock(nn.Module):
    """
    Transformer Block with Causal LinRec Attention.
    """

    def __init__(self, emb_size, n_heads, hidden_size, dropout=0.1):
        super().__init__()
        self.attention = LinRecAttention(emb_size, n_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # Attention -> Residual -> Norm
        att_out = self.attention(x)
        x = self.norm1(x + self.dropout1(att_out))

        # FFN -> Residual -> Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


@model_registry.register(name="LinRec")
class LinRec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of LinRec algorithm from
        "LinRec: Linear Attention Mechanism for Long-term Sequential Recommender Systems" in SIGIR 2023.

    LinRec replaces the quadratic Dot-Product Attention with an O(N) Linear Attention
    mechanism based on L2 Normalization and ELU activation.

    For further details, please refer to the `paper <https://arxiv.org/abs/2411.01537>`_.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): Item embedding dimensions.
        n_layers (int): The number of transformer encoder layers.
        n_heads (int): The number of attention heads in the transformer.
        inner_size (int): The dimensionality of the feed-forward layer in the transformer.
        dropout_prob (float): Dropout probability.
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used during training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
    """

    DATALOADER_TYPE = DataLoaderType.SEQUENTIAL_LOADER

    embedding_size: int
    n_layers: int
    n_heads: int
    inner_size: int
    dropout_prob: float
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

        # Item Embeddings
        self.item_embedding = nn.Embedding(
            self.n_items + 1,
            self.embedding_size,
            padding_idx=self.n_items,
        )

        # Positional Embeddings
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)

        self.emb_dropout = nn.Dropout(self.dropout_prob)

        # Stack of LinRec Blocks (Transformer Layers)
        self.layers = nn.ModuleList(
            [
                LinRecBlock(
                    self.embedding_size,
                    self.n_heads,
                    self.inner_size,
                    self.dropout_prob,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(self.embedding_size)

        self.apply(self._init_weights)

        # Loss function will be based on number of
        # negative samples
        self.main_loss: nn.Module
        if self.neg_samples > 0:
            self.main_loss = BPRLoss()
        else:
            self.main_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

    def get_dataloader(
        self, interactions: Interactions, sessions: Sessions, **kwargs: Any
    ):
        return sessions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
            **kwargs,
        )

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        # item_seq: [B, L]
        batch_size, seq_len = item_seq.shape

        # Create Position IDs: [0, 1, 2, ..., L-1]
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Embedding Lookup: Item + Position
        items_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(position_ids)

        x = self.emb_dropout(items_emb + pos_emb)

        # Pass through stacked Transformer Blocks
        for layer in self.layers:
            x = layer(x)

        # Final Layer Norm
        seq_output = self.layer_norm(x)

        # Gather the representation at the last non-padded index
        # This represents the user's state after the full sequence
        seq_output = self._gather_indexes(seq_output, item_seq_len - 1)

        return seq_output

    def train_step(self, batch: Any, *args, **kwargs):
        if self.neg_samples > 0:
            item_seq, item_seq_len, pos_item, neg_item = batch
        else:
            item_seq, item_seq_len, pos_item = batch
            neg_item = None

        # Standard Sequential slicing: Input is items [0...n-1], Target is [n]
        input_seq = item_seq[:, :-1]
        input_len = torch.clamp(item_seq_len - 1, min=1)

        seq_output = self.forward(input_seq, input_len)
        pos_items_emb = self.item_embedding(pos_item)

        if self.neg_samples > 0:
            neg_items_emb = self.item_embedding(neg_item)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output.unsqueeze(1) * neg_items_emb, dim=-1)
            main_loss = self.main_loss(pos_score, neg_score)

            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(input_seq), pos_items_emb, neg_items_emb
            )
        else:
            # Full Softmax over all items
            test_item_emb = self.item_embedding.weight[:-1, :]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            main_loss = self.main_loss(logits, pos_item)

            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(input_seq), pos_items_emb
            )

        return main_loss + reg_loss

    @torch.no_grad()
    def predict(
        self,
        user_seq: Tensor,
        seq_len: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Prediction using the learned session embeddings.

        Args:
            user_seq (Tensor): Padded sequences of item IDs for users to predict for.
            seq_len (Tensor): Actual lengths of these sequences, before padding.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
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
