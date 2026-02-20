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


@model_registry.register(name="NARM")
class NARM(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of NARM algorithm from
    "Neural Attentive Session-based Recommendation." in CIKM 2017.

    NARM explores a hybrid encoder with an attention mechanism to model the
    user’s sequential behavior (Global Encoder) and capture the user’s
    main purpose in the current session (Local Encoder).

    For further details, please refer to the `paper <https://arxiv.org/abs/1711.04725>`_.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The dimension of the item embeddings.
        hidden_size (int): The number of features in the hidden state of the GRU.
        n_layers (int): The number of recurrent layers in the GRU.
        hidden_dropout_prob (float): Dropout probability for the item embeddings.
        attn_dropout_prob (float): Dropout probability for the hybrid session representation.
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.SEQUENTIAL_LOADER

    # Model hyperparameters
    embedding_size: int
    hidden_size: int
    n_layers: int
    hidden_dropout_prob: float
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

        # Item embedding
        self.item_embedding = nn.Embedding(
            self.n_items + 1,
            self.embedding_size,
            padding_idx=self.n_items,
        )
        self.emb_dropout = nn.Dropout(self.hidden_dropout_prob)

        # Sequential Encoder (GRU)
        self.gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bias=False,
            batch_first=True,
        )

        # Attention layers for Local Encoder
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.attn_dropout_prob)

        # Final projection to align hybrid representation with item embedding space
        self.b = nn.Linear(2 * self.hidden_size, self.embedding_size, bias=False)

        # Initialize weights using Xavier Normal (recommended in NARM paper)
        self.apply(self._init_weights)

        # Loss function setup
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
        **kwargs: Any,
    ):
        return sessions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
            **kwargs,
        )

    def train_step(self, batch: Any, *args, **kwargs):
        if self.neg_samples > 0:
            item_seq, item_seq_len, pos_item, neg_item = batch
        else:
            item_seq, item_seq_len, pos_item = batch
            neg_item = None

        seq_output = self.forward(item_seq, item_seq_len)
        pos_items_emb = self.item_embedding(pos_item)

        if self.neg_samples > 0:
            # Pairwise BPR Loss
            neg_items_emb = self.item_embedding(neg_item)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output.unsqueeze(1) * neg_items_emb, dim=-1)
            main_loss = self.main_loss(pos_score, neg_score)

            # L2 Regularization
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq),
                pos_items_emb,
                neg_items_emb,
            )
        else:
            # Pointwise Cross Entropy Loss
            test_item_emb = self.item_embedding.weight[:-1, :]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            main_loss = self.main_loss(logits, pos_item)

            # L2 Regularization
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq),
                pos_items_emb,
            )

        return main_loss + reg_loss

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        """Forward pass of the NARM model.

        Args:
            item_seq (Tensor): Padded sequences of item IDs [batch_size, max_seq_len].
            item_seq_len (Tensor): Actual lengths of sequences.

        Returns:
            Tensor: The hybrid session representation [batch_size, embedding_size].
        """
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)

        # GRU encoding
        gru_out, _ = self.gru(item_seq_emb_dropout)

        # Global Encoder (c_global): The last hidden state of the GRU
        c_global = self._gather_indexes(gru_out, item_seq_len - 1)

        # Local Encoder (c_local): Attention mechanism over all hidden states
        # Avoid influence of padding tokens in attention
        mask = (item_seq != self.n_items).unsqueeze(2).expand_as(gru_out)
        q1 = self.a_1(gru_out)
        q2 = self.a_2(c_global).unsqueeze(1)

        # Calculate attention weights alpha

        # Compute non-normalized logits
        # q1: [B, S, H]
        # q2: [B, 1, H]
        # v_t output: [B, S, 1]
        alpha_logits = self.v_t(torch.sigmoid(q1 + q2))

        # Apply masking: [B, S, 1]
        mask = (item_seq != self.n_items).unsqueeze(2)
        alpha_logits = alpha_logits.masked_fill(mask == 0, -1e9)

        # Apply softmax over temporal dimension
        alpha = F.softmax(alpha_logits, dim=1)
        c_local = torch.sum(alpha * gru_out, dim=1)  # Weighted sum

        # Hybrid Representation: Concatenate Local and Global vectors
        c_t = torch.cat([c_local, c_global], dim=1)
        c_t = self.ct_dropout(c_t)

        # Final projection to the item embedding space
        seq_output = self.b(c_t)

        return seq_output

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
