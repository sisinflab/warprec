# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import EmbLoss
from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class SampledSoftmaxLoss(nn.Module):
    """Cross-entropy over one positive item and sampled negatives."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, seq_output: Tensor, pos_item_emb: Tensor, neg_item_emb: Tensor
    ) -> Tensor:
        pos_score = torch.sum(seq_output * pos_item_emb, dim=-1, keepdim=True)
        neg_score = torch.matmul(seq_output.unsqueeze(1), neg_item_emb.transpose(1, 2))
        neg_score = neg_score.squeeze(1)

        logits = torch.cat([pos_score, neg_score], dim=1) / self.temperature
        targets = torch.zeros(
            seq_output.size(0), dtype=torch.long, device=seq_output.device
        )
        return F.cross_entropy(logits, targets)


class LiGRBlock(nn.Module):
    """LiGR Transformer block with gated attention and gated feed-forward."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_dropout: float,
        dropout: float,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-8)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-8)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_gate = nn.Linear(d_model, d_model)

        # SwiGLU feed-forward, as described in the eSASRec paper.
        self.ff_value = nn.Linear(d_model, d_ff)
        self.ff_gate_hidden = nn.Linear(d_model, d_ff)
        self.ff_out = nn.Linear(d_ff, d_model)
        self.ff_gate = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        attn_input = self.attn_norm(hidden_states)
        attn_output, _ = self.attention(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
            is_causal=True,
        )
        attn_output = attn_output * torch.sigmoid(self.attn_gate(attn_input))
        hidden_states = hidden_states + self.dropout(attn_output)

        ffn_input = self.ffn_norm(hidden_states)
        ff_hidden = F.silu(self.ff_value(ffn_input)) * self.ff_gate_hidden(ffn_input)
        ff_output = self.ff_out(ff_hidden)
        ff_output = ff_output * torch.sigmoid(self.ff_gate(ffn_input))
        hidden_states = hidden_states + self.dropout(ff_output)
        return hidden_states


@model_registry.register(name="eSASRec")
class eSASRec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of eSASRec from
    "eSASRec: Enhancing Transformer-based Recommendations in a Modular Fashion."

    The model is built around the winning combination described in the paper:
    shifted-sequence objective, LiGR Transformer blocks, and sampled softmax,
    with optional mixed negative sampling.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): Dimension of item and position embeddings.
        n_layers (int): Number of transformer encoder layers.
        n_heads (int): Number of attention heads in the transformer.
        inner_size (int): Dimension of the feedforward network in the transformer.
        dropout_prob (float): Dropout probability for embeddings.
        attn_dropout_prob (float): Dropout probability for attention weights.
        use_relative_pos (bool): Whether to use relative positional embeddings.
        use_sampled_softmax (bool): Whether to use sampled softmax loss.
        use_ligr (bool): Whether to use LiGR blocks instead of standard transformer layers.
        mn_ratio (float): Ratio of in-batch negatives to uniform negatives when using mixed negative sampling.
        reg_weight (float): Weight for the embedding regularization loss.
        weight_decay (float): L2 regularization weight for optimizer.
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        neg_samples (int): Number of negative samples for training.
        max_seq_len (int): Maximum length of input sequences.
    """

    DATALOADER_TYPE = DataLoaderType.SEQUENTIAL_LOADER

    embedding_size: int
    n_layers: int
    n_heads: int
    inner_size: int
    dropout_prob: float
    attn_dropout_prob: float
    use_relative_pos: bool
    use_sampled_softmax: bool
    use_ligr: bool
    mn_ratio: float
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
        # eSASRec keeps SASRec's absolute positions for the shifted-sequence objective.
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)

        if self.use_ligr:
            self.encoder = nn.ModuleList(
                [
                    LiGRBlock(
                        d_model=self.embedding_size,
                        n_heads=self.n_heads,
                        d_ff=self.inner_size,
                        attn_dropout=self.attn_dropout_prob,
                        dropout=self.dropout_prob,
                    )
                    for _ in range(self.n_layers)
                ]
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_size,
                nhead=self.n_heads,
                dim_feedforward=self.inner_size,
                dropout=self.attn_dropout_prob,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.n_layers,
            )

        causal_mask = self._generate_square_subsequent_mask(self.max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

        self.apply(self._init_weights)

        self.full_softmax_loss = nn.CrossEntropyLoss()
        self.sampled_softmax_loss = SampledSoftmaxLoss(temperature=1.0)
        self.reg_loss = EmbLoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs,
    ):
        return sessions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
            **kwargs,
        )

    def _mix_negative_items(
        self, pos_item: Tensor, neg_item: Optional[Tensor]
    ) -> Optional[Tensor]:
        """Mix uniform negatives with in-batch negatives when requested."""
        if (
            neg_item is None
            or self.neg_samples <= 0
            or self.mn_ratio <= 0
            or pos_item.size(0) <= 1
        ):
            return neg_item

        num_in_batch = int(round(self.neg_samples * self.mn_ratio))
        num_in_batch = max(0, min(self.neg_samples, num_in_batch))
        if num_in_batch == 0:
            return neg_item

        num_uniform = self.neg_samples - num_in_batch
        uniform_part = (
            neg_item[:, :num_uniform]
            if num_uniform > 0
            else torch.empty(
                pos_item.size(0),
                0,
                dtype=pos_item.dtype,
                device=pos_item.device,
            )
        )

        batch_size = pos_item.size(0)

        # Create a boolean candidate mask of shape [B, B]
        # candidate_mask[i, j] is True if pos_item[j] != pos_item[i]
        # This naturally avoids self-negation (j == i) and duplicate items in the batch
        candidate_mask = pos_item.unsqueeze(1) != pos_item.unsqueeze(0)

        # Assign random weights to all batch elements to perform uniform sampling
        rand_weights = torch.rand(batch_size, batch_size, device=pos_item.device)

        # Zero out weights for invalid candidates
        rand_weights = torch.where(
            candidate_mask, rand_weights, torch.zeros_like(rand_weights)
        )

        # Retrieve the top-k indices with the largest weights
        sampled_vals, sampled_idx = torch.topk(rand_weights, num_in_batch, dim=1)

        # Handle potential edge cases where a row has fewer valid candidates than num_in_batch
        # Create a deterministic fallback index matrix (e.g., shifting indices safely)
        row_indices = (
            torch.arange(batch_size, device=pos_item.device)
            .unsqueeze(1)
            .expand(-1, num_in_batch)
        )
        shift_offsets = torch.arange(
            1, num_in_batch + 1, device=pos_item.device
        ).unsqueeze(0)
        fallback_idx = (row_indices + shift_offsets) % batch_size

        # Check if the sampled top-k items are actually valid (weight > 0)
        is_valid = sampled_vals > 0.0
        final_idx = torch.where(is_valid, sampled_idx, fallback_idx)

        # Gather the final mixed in-batch negatives
        in_batch_part = pos_item[final_idx]

        return torch.cat([uniform_part, in_batch_part], dim=1)

    def _compute_main_loss(
        self, seq_output: Tensor, pos_item: Tensor, neg_item: Optional[Tensor]
    ) -> Tensor:
        if self.use_sampled_softmax and neg_item is not None and neg_item.size(1) > 0:
            pos_items_emb = self.item_embedding(pos_item)
            neg_items_emb = self.item_embedding(neg_item)
            return self.sampled_softmax_loss(seq_output, pos_items_emb, neg_items_emb)

        item_embeddings = self.item_embedding.weight[:-1, :]
        logits = torch.matmul(seq_output, item_embeddings.transpose(0, 1))
        return self.full_softmax_loss(logits, pos_item)

    def training_step(self, batch: Any, batch_idx: int):
        if len(batch) == 4:
            item_seq, item_seq_len, pos_item, neg_item = batch
        else:
            item_seq, item_seq_len, pos_item = batch
            neg_item = None

        neg_item = self._mix_negative_items(pos_item, neg_item)
        seq_output = self.forward(item_seq, item_seq_len)
        main_loss = self._compute_main_loss(seq_output, pos_item, neg_item)

        reg_terms = [self.item_embedding(item_seq), self.item_embedding(pos_item)]
        if neg_item is not None:
            reg_terms.append(self.item_embedding(neg_item))
        reg_loss = self.reg_weight * self.reg_loss(*reg_terms)

        return main_loss + reg_loss

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        """Forward pass with shifted-sequence causal masking."""
        seq_len = item_seq.size(1)
        padding_mask = item_seq == self.n_items

        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(position_ids)
        seq_emb = self.layernorm(item_emb + pos_emb)
        seq_emb = self.emb_dropout(seq_emb)

        attn_mask = self.causal_mask[:seq_len, :seq_len]  # type: ignore[index]

        if self.use_ligr:
            output = seq_emb
            for layer in self.encoder:
                output = layer(output, attn_mask, padding_mask)
            transformer_output = output
        else:
            transformer_output = self.transformer_encoder(
                src=seq_emb,
                mask=attn_mask,
                src_key_padding_mask=padding_mask,
            )

        return self._gather_indexes(transformer_output, item_seq_len - 1)

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
        """Prediction using the learned session embeddings."""
        seq_output = self.forward(user_seq, seq_len)

        if item_indices is None:
            item_embeddings = self.item_embedding.weight[:-1, :]
            einsum_string = "be,ie->bi"
        else:
            item_embeddings = self.item_embedding(item_indices)
            einsum_string = "be,bse->bs"

        return torch.einsum(einsum_string, seq_output, item_embeddings)
