# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235, R0902
from typing import Callable, Any, Optional

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


@model_registry.register(name="gSASRec")
class gSASRec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of gSASRec (generalized SASRec).

    This model adapts the SASRec architecture to predict the next item at every
    step of the sequence, using a Group-wise Binary Cross-Entropy (GBCE) loss function.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
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
        gbce_t (float): The temperature parameter for the Group-wise Binary Cross-Entropy loss.
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
        reuse_item_embeddings (bool): Whether to reuse item embeddings for output or not.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.USER_HISTORY_LOADER

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
    gbce_t: float
    neg_samples: int
    max_seq_len: int
    reuse_item_embeddings: bool

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, info, *args, seed=seed, **kwargs)

        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)

        if not self.reuse_item_embeddings:
            self.output_embedding = nn.Embedding(
                self.n_items + 1, self.embedding_size, padding_idx=self.n_items
            )

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)
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

        # Precompute causal mask
        causal_mask = self._generate_square_subsequent_mask(self.max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

        # Initialize weights
        self.apply(self._init_weights)
        self.gbce_loss = self._gbce_loss_function()
        self.reg_loss = EmbLoss()

    def _get_output_embeddings(self) -> nn.Embedding:
        """Return embeddings based on the flag value reuse_item_embeddings.

        Returns:
            nn.Embedding: The item embedding if reuse_item_embeddings is True,
                else the output embedding.
        """
        if self.reuse_item_embeddings:
            return self.item_embedding
        return self.output_embedding

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return sessions.get_user_history_dataloader(
            max_seq_len=self.max_seq_len,
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
            low_memory=low_memory,
        )

    def forward(self, item_seq: Tensor) -> Tensor:
        """Forward pass of gSASRec. Returns the output of the Transformer
        for each token in the input sequence.

        Args:
            item_seq (Tensor): Sequence of items [batch_size, seq_len].

        Returns:
            Tensor: Output of the Transformer encoder [batch_size, seq_len, embedding_size].
        """
        seq_len = item_seq.size(1)
        padding_mask = item_seq == self.n_items

        position_ids = torch.arange(seq_len, dtype=torch.long).to(item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(position_ids)

        seq_emb = self.layernorm(item_emb + pos_emb)
        seq_emb = self.emb_dropout(seq_emb)

        transformer_output = self.transformer_encoder(
            src=seq_emb,
            mask=self.causal_mask[:seq_len, :seq_len],  # type:ignore [index]
            src_key_padding_mask=padding_mask,
        )
        return transformer_output

    def _gbce_loss_function(self) -> Callable:
        """Return the General Binary Cross-Entropy (GBCE) loss.

        Returns:
            Callable: The GBCE loss.
        """

        def gbce_loss_fn(
            sequence_hidden_states: Tensor,
            labels: Tensor,
            negatives: Tensor,
            model_input: Tensor,
        ):
            pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives], dim=-1)
            pos_neg_embeddings = self._get_output_embeddings()(pos_neg_concat)

            logits = torch.einsum(
                "bse, bsne -> bsn", sequence_hidden_states, pos_neg_embeddings
            )

            gt = torch.zeros_like(logits).to(logits.device)
            gt[:, :, 0] = 1.0

            alpha = self.neg_samples / (self.n_items - 1)
            t = self.gbce_t
            beta = alpha * ((1 - 1 / alpha) * t + 1 / alpha)

            positive_logits = logits[:, :, 0:1].to(torch.float64)
            negative_logits = logits[:, :, 1:].to(torch.float64)
            eps = 1e-10

            positive_probs = torch.clamp(torch.sigmoid(positive_logits), eps, 1 - eps)
            positive_probs_pow = torch.clamp(
                positive_probs.pow(-beta),
                min=1.0 + eps,
                max=torch.finfo(torch.float64).max,
            )
            to_log = torch.clamp(
                torch.div(1.0, (positive_probs_pow - 1)),
                eps,
                torch.finfo(torch.float64).max,
            )
            positive_logits_transformed = to_log.log()

            final_logits = torch.cat(
                [positive_logits_transformed, negative_logits], -1
            ).to(torch.float32)

            mask = (labels != self.n_items).float()
            loss_per_element = nn.functional.binary_cross_entropy_with_logits(
                final_logits, gt, reduction="none"
            )

            loss_per_element = loss_per_element.mean(-1) * mask
            total_loss = loss_per_element.sum() / mask.sum().clamp(min=1)
            return total_loss

        return gbce_loss_fn

    def train_step(self, batch: Any, *args, **kwargs):
        positives, negatives = batch

        if positives.shape[0] == 0 or positives.shape[1] < 2:
            return torch.tensor(0.0, requires_grad=True).to(positives.device)

        model_input = positives[:, :-1]
        labels = positives[:, 1:]

        if model_input.shape[1] == 0:
            return torch.tensor(0.0, requires_grad=True).to(positives.device)

        # Calculate GBCE loss
        sequence_hidden_states = self.forward(model_input)
        gbce_loss = self.gbce_loss(
            sequence_hidden_states, labels, negatives, model_input
        )

        # Calculate L2 regularization
        reg_loss = self.reg_weight * self.reg_loss(
            self.item_embedding(model_input),
            self._get_output_embeddings()(labels),
            self._get_output_embeddings()(negatives),
        )

        return gbce_loss + reg_loss

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
        # Get the transformer output for all tokens in the sequence
        transformer_output = self.forward(user_seq)

        # Get the embedding of the LAST relevant item for prediction
        seq_output = self._gather_indexes(
            transformer_output, seq_len - 1
        )  # [batch_size, embedding_size]

        target_item_embeddings = self._get_output_embeddings()
        if item_indices is None:
            # Case 'full': prediction on all items
            item_embeddings = target_item_embeddings.weight[
                :-1, :
            ]  # [n_items, embedding_size]
            einsum_string = "be,ie->bi"  # b: batch, e: embedding, i: item
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = target_item_embeddings(
                item_indices
            )  # [batch_size, pad_seq, embedding_size]
            einsum_string = "be,bse->bs"  # b: batch, e: embedding, s: sample

        predictions = torch.einsum(
            einsum_string, seq_output, item_embeddings
        )  # [batch_size, n_items] or [batch_size, pad_seq]
        return predictions
