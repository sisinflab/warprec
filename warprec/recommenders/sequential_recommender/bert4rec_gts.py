# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235, R0902
from typing import Any, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry

@model_registry.register(name="BERT4Rec_gts")
class BERT4Rec_gts(IterativeRecommender, SequentialRecommenderUtils):
    
    DATALOADER_TYPE = DataLoaderType.CLOZE_MASK_LOADER

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        # --- FIX COMPLETO: ASSEGNAZIONE DI TUTTI I PARAMETRI ---
        self.embedding_size = params.get("embedding_size")
        self.n_layers = params.get("n_layers")
        self.n_heads = params.get("n_heads")
        self.inner_size = params.get("inner_size")
        self.dropout_prob = params.get("dropout_prob")
        self.attn_dropout_prob = params.get("attn_dropout_prob")
        self.mask_prob = params.get("mask_prob")
        self.reg_weight = params.get("reg_weight")
        self.weight_decay = params.get("weight_decay")
        self.batch_size = params.get("batch_size")
        self.epochs = params.get("epochs")
        self.learning_rate = params.get("learning_rate")
        self.neg_samples = params.get("neg_samples")
        self.max_seq_len = params.get("max_seq_len")
        # -------------------------------------------------------

        # Definizione ID speciali
        self.padding_token_id = self.n_items
        self.mask_token_id = self.n_items + 1

        # Item embedding
        self.item_embedding = nn.Embedding(
            self.n_items + 2, self.embedding_size, padding_idx=self.padding_token_id
        )

        # Position embedding
        self.position_embedding = nn.Embedding(
            self.max_seq_len + 1, self.embedding_size
        )
        
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)
        self.dropout = nn.Dropout(self.dropout_prob)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.attn_dropout_prob,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        # Testa di predizione (Full Evaluation / Cross-Entropy)
        self.output_layer = nn.Linear(self.embedding_size, self.n_items)
        
        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def get_dataloader(self, interactions: Interactions, sessions: Sessions, **kwargs):
        return sessions.get_cloze_mask_dataloader(
            max_seq_len=self.max_seq_len,
            mask_prob=self.mask_prob,
            neg_samples=self.neg_samples, # Ora self.neg_samples esiste
            batch_size=self.batch_size,   # Ora self.batch_size esiste
            mask_token_id=self.mask_token_id,
            **kwargs,
        )

    def forward(self, item_seq: Tensor) -> Tensor:
        padding_mask = (item_seq == self.padding_token_id)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        ).unsqueeze(0).expand_as(item_seq)

        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(position_ids)

        input_emb = self.layernorm(item_emb + pos_emb)
        input_emb = self.dropout(input_emb)

        transformer_output = self.transformer_encoder(
            src=input_emb, src_key_padding_mask=padding_mask
        )
        return transformer_output

    def train_step(self, batch: Any, *args, **kwargs):
        masked_seq, pos_items, _, masked_indices = batch

        transformer_output = self.forward(masked_seq)
        masked_output = self._multi_hot_gather(transformer_output, masked_indices)
        logits = self.output_layer(masked_output)

        valid_mask = (masked_indices > 0)
        flat_logits = logits[valid_mask]
        flat_targets = pos_items[valid_mask]

        return self.loss_fct(flat_logits, flat_targets)

    def _multi_hot_gather(self, source: Tensor, indices: Tensor) -> Tensor:
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, source.size(-1))
        return torch.gather(source, 1, indices_expanded)

    def _prepare_for_prediction(self, user_seq: Tensor, seq_len: Tensor) -> Tensor:
        pred_seq = torch.full(
            (user_seq.size(0), user_seq.size(1) + 1),
            self.padding_token_id,
            dtype=torch.long,
            device=user_seq.device,
        )
        pred_seq[:, : user_seq.size(1)] = user_seq
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
        pred_seq = self._prepare_for_prediction(user_seq, seq_len)
        transformer_output = self.forward(pred_seq)
        seq_output = self._gather_indexes(transformer_output, seq_len)
        logits = self.output_layer(seq_output)

        if item_indices is not None:
            logits = torch.gather(logits, 1, item_indices)

        return logits