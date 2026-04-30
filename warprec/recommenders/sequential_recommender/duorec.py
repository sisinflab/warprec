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


@model_registry.register(name="DuoRec")
class DuoRec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of DuoRec model
    "Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation" in WSDM 2022.

    DuoRec extends a SASRec-style backbone with two contrastive regularizers:
    1. Unsupervised CL from two stochastic forward passes of the same sequence.
    2. Supervised CL from another sequence sharing the same next-item target.

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
        ssl_type (str): Type of self-supervised learning ("us", "su", "un", "us_x").
        ssl_lambda (float): Weight for the unsupervised CL loss.
        ssl_lambda_sem (float): Weight for the supervised CL loss.
        tau (float): Temperature parameter for contrastive loss.
        sim_type (str): Similarity metric for contrastive loss ("dot" or "cos").
        reg_weight (float): Weight for the embedding regularization loss.
        weight_decay (float): L2 regularization weight for optimizer.
        batch_size (int): Training batch size.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
        neg_samples (int): Number of negative samples for training.
        max_seq_len (int): Maximum length of input sequences.
    """

    DATALOADER_TYPE = DataLoaderType.SEQUENTIAL_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    n_heads: int
    inner_size: int
    dropout_prob: float
    attn_dropout_prob: float
    ssl_type: str
    ssl_lambda: float
    ssl_lambda_sem: float
    tau: float
    sim_type: str
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
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

        causal_mask = self._generate_square_subsequent_mask(self.max_seq_len)
        self.register_buffer("causal_mask", causal_mask)
        self.register_buffer(
            "position_ids", torch.arange(self.max_seq_len, dtype=torch.long)
        )

        self.apply(self._init_weights)

        self.main_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return sessions.get_same_target_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            batch_size=self.batch_size,
            low_memory=low_memory,
            **kwargs,
        )

    def training_step(self, batch: Any, batch_idx: int):
        item_seq, item_seq_len, pos_item, sem_seq, sem_seq_len, has_sem_pos = batch

        seq_output = self.forward(item_seq, item_seq_len)
        logits = torch.matmul(
            seq_output, self.item_embedding.weight[:-1].transpose(0, 1)
        )
        main_loss = self.main_loss(logits, pos_item)
        reg_loss = self.reg_weight * self.reg_loss(
            self.item_embedding(item_seq),
            self.item_embedding(pos_item),
        )
        total_loss = main_loss + reg_loss

        ssl_mode = self.ssl_type.lower()

        if ssl_mode in {"us", "un", "us_x"} and self.ssl_lambda > 0:
            aug_seq_output = self.forward(item_seq, item_seq_len)
            total_loss += self.ssl_lambda * self._contrastive_loss(
                seq_output, aug_seq_output, pos_item
            )

        if ssl_mode in {"su", "us_x"} and self.ssl_lambda_sem > 0:
            valid_sem_mask = has_sem_pos.bool()
            if valid_sem_mask.any():
                sem_seq_output = self.forward(
                    sem_seq[valid_sem_mask], sem_seq_len[valid_sem_mask]
                )
                total_loss += self.ssl_lambda_sem * self._contrastive_loss(
                    seq_output[valid_sem_mask],
                    sem_seq_output,
                    pos_item[valid_sem_mask],
                )

        return total_loss

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        """Encode the sequence and return the final valid hidden state."""
        seq_len = item_seq.size(1)
        position_ids = self.position_ids[:seq_len].unsqueeze(0).expand_as(item_seq)  # type: ignore[index]

        item_emb = self.item_embedding(item_seq)
        position_emb = self.position_embedding(position_ids)
        seq_emb = self.layernorm(item_emb + position_emb)
        seq_emb = self.emb_dropout(seq_emb)

        padding_mask = item_seq == self.n_items
        attention_mask = self.causal_mask[:seq_len, :seq_len]  # type: ignore[index]
        transformer_output = self.transformer_encoder(
            src=seq_emb,
            mask=attention_mask,
            src_key_padding_mask=padding_mask,
            is_causal=True,
        )

        return self._gather_indexes(transformer_output, item_seq_len - 1)

    def _contrastive_loss(
        self, z_i: Tensor, z_j: Tensor, target_items: Tensor
    ) -> Tensor:
        """InfoNCE with same-target samples removed from the negative pool."""
        batch_size = z_i.size(0)
        representations = torch.cat((z_i, z_j), dim=0)

        if self.sim_type == "cos":
            representations = F.normalize(representations, dim=1)
        elif self.sim_type != "dot":
            raise ValueError(f"Unknown similarity metric: {self.sim_type}")

        sim_matrix = torch.matmul(representations, representations.transpose(0, 1))
        sim_matrix = sim_matrix / self.tau

        targets = torch.cat((target_items, target_items), dim=0)
        total = 2 * batch_size
        row_indices = torch.arange(total, device=sim_matrix.device)
        positive_indices = (row_indices + batch_size) % total

        negative_mask = torch.ones(
            (total, total), dtype=torch.bool, device=sim_matrix.device
        )
        negative_mask.fill_diagonal_(False)
        negative_mask[row_indices, positive_indices] = False

        same_target_mask = targets.unsqueeze(0).eq(targets.unsqueeze(1))
        negative_mask &= ~same_target_mask

        logits = sim_matrix.masked_fill(~negative_mask, -1e9)
        logits[row_indices, positive_indices] = sim_matrix[
            row_indices, positive_indices
        ]

        return F.cross_entropy(logits, positive_indices)

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
        """Prediction using the learned sequence embeddings."""
        seq_output = self.forward(user_seq, seq_len)

        if item_indices is None:
            item_embeddings = self.item_embedding.weight[:-1, :]
            einsum_string = "be,ie->bi"
        else:
            item_embeddings = self.item_embedding(item_indices)
            einsum_string = "be,bse->bs"

        return torch.einsum(einsum_string, seq_output, item_embeddings)
