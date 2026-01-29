# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
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


@model_registry.register(name="SASRec")
class SASRec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of SASRec algorithm from
    "Self-Attentive Sequential Recommendation." in ICDM 2018.

    This implementation is adapted to the WarpRec framework, using PyTorch's
    native nn.TransformerEncoder for the self-attention mechanism.

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

        # Precompute causal mask
        causal_mask = self._generate_square_subsequent_mask(self.max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

        # Initialize weights
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
            main_loss = self.main_loss(pos_score, neg_score)

            # L2 regularization
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq),
                self.item_embedding(pos_item),
                self.item_embedding(neg_item),
            )
        else:
            test_item_emb = self.item_embedding.weight  # [n_items, embedding_size]
            logits = torch.matmul(
                seq_output, test_item_emb.transpose(0, 1)
            )  # [batch_size, n_items]
            main_loss = self.main_loss(logits, pos_item)

            # L2 regularization
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq),
                self.item_embedding(pos_item),
            )

        return main_loss + reg_loss

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
        padding_mask = item_seq == self.n_items  # [batch_size, seq_len]

        # Create position IDs
        position_ids = torch.arange(seq_len, dtype=torch.long).to(item_seq.device)
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
            mask=self.causal_mask,
            src_key_padding_mask=padding_mask,
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
