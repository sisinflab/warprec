# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235, R0902
from typing import Any, Optional

import torch
from torch import nn, Tensor

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="BERT4Rec")
class BERT4Rec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of BERT4Rec algorithm from
    "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer."
    in CIKM 2019.

    This model uses a bidirectional Transformer to learn item representations based on a
    masked item prediction task (cloze task). For next-item prediction, a special [MASK]
    token is appended to the sequence.

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
        inner_size (int): The dimensionality of the feed-forward layer in the transformer.
        dropout_prob (float): The probability of dropout for embeddings and other layers.
        attn_dropout_prob (float): The probability of dropout for the attention weights.
        mask_prob (float): The probability of an item being masked during training.
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used during training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples for BPR loss.
        max_seq_len (int): The maximum length of sequences.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.CLOZE_MASK_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    n_heads: int
    inner_size: int
    dropout_prob: float
    attn_dropout_prob: float
    mask_prob: float
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

        # Define special token IDs
        self.padding_token_id = self.n_items
        self.mask_token_id = self.n_items + 1

        # Item embedding needs to accommodate items, padding token, and mask token
        self.item_embedding = nn.Embedding(
            self.n_items + 2, self.embedding_size, padding_idx=self.padding_token_id
        )

        # Take into account the extra [MASK] token in position embeddings
        self.position_embedding = nn.Embedding(
            self.max_seq_len + 1, self.embedding_size
        )
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)
        self.dropout = nn.Dropout(self.dropout_prob)

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
            encoder_layer, num_layers=self.n_layers
        )

        # Final projection layer, as in the original implementation
        self.out_bias = nn.Parameter(torch.zeros(self.n_items + 1))

        self.apply(self._init_weights)
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return sessions.get_cloze_mask_dataloader(
            max_seq_len=self.max_seq_len,
            mask_prob=self.mask_prob,
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
            mask_token_id=self.mask_token_id,
            low_memory=low_memory,
        )

    def train_step(self, batch: Any, *args, **kwargs):
        masked_seq, pos_items, neg_items, masked_indices = batch

        # Get the output of the bidirectional transformer
        transformer_output = self.forward(masked_seq)

        # Gather the hidden states at the masked positions
        seq_output = self._multi_hot_gather(transformer_output, masked_indices)

        # Get embeddings for positive and negative items
        pos_items_emb = self.item_embedding(pos_items)
        neg_items_emb = self.item_embedding(neg_items)

        # Get the output bias for positive and negative items
        pos_bias = self.out_bias[pos_items]
        neg_bias = self.out_bias[neg_items]

        # Calculate BPR loss
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1) + pos_bias
        neg_score = (
            torch.sum(seq_output.unsqueeze(2) * neg_items_emb, dim=-1) + neg_bias
        )
        loss_mask = masked_indices > 0
        bpr_loss = self.bpr_loss(pos_score[loss_mask], neg_score[loss_mask])

        # Calculate L2 regularization
        reg_loss = self.reg_weight * self.reg_loss(
            self.item_embedding(masked_seq),
            pos_items_emb,
            neg_items_emb,
        )

        return bpr_loss + reg_loss

    def forward(self, item_seq: Tensor) -> Tensor:
        """
        Forward pass of BERT4Rec. Uses bidirectional attention.

        Args:
            item_seq (Tensor): Sequence of items, potentially with [MASK] tokens.

        Returns:
            Tensor: Output of the Transformer for each token [batch_size, seq_len, embedding_size].
        """
        # Padding mask to ignore padding tokens
        padding_mask = item_seq == self.padding_token_id

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(position_ids)

        input_emb = self.layernorm(item_emb + pos_emb)
        input_emb = self.dropout(input_emb)

        # For bidirectional attention, the causal mask is None
        transformer_output = self.transformer_encoder(
            src=input_emb, mask=None, src_key_padding_mask=padding_mask
        )
        return transformer_output

    def _multi_hot_gather(self, source: Tensor, indices: Tensor) -> Tensor:
        """Gathers specific vectors from a source tensor based on indices.
        This is an efficient way to select the transformer outputs at masked positions.

        Args:
            source (Tensor): The source tensor [batch_size, seq_len, embedding_size].
            indices (Tensor): The indices to gather [batch_size, num_masked].

        Returns:
            Tensor: The gathered vectors [batch_size, num_masked, embedding_size].
        """
        # Add a dimension for the embedding size
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, source.size(-1))
        return torch.gather(source, 1, indices_expanded)

    def _prepare_for_prediction(self, user_seq: Tensor, seq_len: Tensor) -> Tensor:
        """Appends a [MASK] token at the end of each sequence for next-item prediction."""
        # Create a new sequence with one extra spot for the mask token
        pred_seq = torch.full(
            (user_seq.size(0), user_seq.size(1) + 1),
            self.padding_token_id,
            dtype=torch.long,
            device=user_seq.device,
        )
        pred_seq[:, : user_seq.size(1)] = user_seq

        # Place the mask token at the end of the actual sequence length
        batch_indices = torch.arange(user_seq.size(0), device=user_seq.device)
        pred_seq[batch_indices, seq_len] = self.mask_token_id

        return pred_seq

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
        Prediction using the learned bidirectional embeddings.

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
        # Prepare the sequence by appending a [MASK] token
        pred_seq = self._prepare_for_prediction(user_seq, seq_len)

        # Get the output of the bidirectional transformer
        transformer_output = self.forward(pred_seq)

        # Gather the output embedding at the position of the [MASK] token
        seq_output = self._gather_indexes(
            transformer_output, seq_len
        )  # [batch_size, embedding_size]

        if item_indices is None:
            # Case 'full': use all item embeddings (excluding padding and mask)
            item_embeddings = self.item_embedding.weight[
                : self.n_items, :
            ]  # [n_items, embedding_size]
            einsum_string = "be,ie->bi"  # b: batch, e: embedding, i: item
            bias = self.out_bias[: self.n_items]
        else:
            # Case 'sampled': use only the provided item indices
            item_embeddings = self.item_embedding(
                item_indices
            )  # [batch_size, pad_seq, embedding_size]
            einsum_string = "be,bse->bs"  # b: batch, e: embedding, s: sample
            bias = self.out_bias[item_indices]

        predictions = (
            torch.einsum(einsum_string, seq_output, item_embeddings) + bias
        )  # [batch_size, n_items] or [batch_size, pad_seq]
        return predictions
