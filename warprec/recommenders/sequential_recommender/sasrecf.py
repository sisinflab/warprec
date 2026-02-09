
# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235

from typing import Any, Optional, Dict

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


@model_registry.register(name="SASRecF")
class SASRecF(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of SASRecF algorithm.
    
    SASRecF is an extension of SASRec that concatenates item representations 
    and item attribute representations (features) as the input to the 
    self-attention mechanism.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The dimension of the item and feature embeddings.
        n_layers (int): The number of transformer encoder layers.
        n_heads (int): The number of attention heads.
        inner_size (int): The dimensionality of the feed-forward layer.
        dropout_prob (float): The probability of dropout for embeddings.
        attn_dropout_prob (float): The probability of dropout for attention.
        hidden_act (str): Activation function for the transformer.
        layer_norm_eps (float): Epsilon for layer normalization.
        initializer_range (float): Range for weight initialization.
        selected_features (list): List of feature fields to include.
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

    # Model hyperparameters (mapped from YAML)
    embedding_size: int
    n_layers: int
    n_heads: int
    inner_size: int
    hidden_dropout_prob: float
    attn_dropout_prob: float
    hidden_act: str
    layer_norm_eps: float
    initializer_range: float
    selected_features: list
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
        interactions: Interactions,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, info, *args, seed=seed, **kwargs)

        # Item ID embedding
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )
        
        # Position embedding
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)

        # Feature embeddings: Create an embedding layer for each selected feature
        # We assume categorical features; WarpRec provides feature metadata in 'info'
        self.feature_embeddings = nn.ModuleDict()
        for feature_name in self.selected_features:
            # Get number of unique values for this feature from info
            num_values = info['features_info'][feature_name]['num_values']
            self.feature_embeddings[feature_name] = nn.Embedding(
                num_values + 1, self.embedding_size, padding_idx=num_values
            )

        # Concatenation layer: maps [Item_Emb + Sum(Feature_Embs)] back to embedding_size
        # Total fields = 1 (Item ID) + number of selected features
        total_fields = 1 + len(self.selected_features)
        self.concat_layer = nn.Linear(
            self.embedding_size * total_fields, self.embedding_size
        )

        self.layernorm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.emb_dropout = nn.Dropout(self.hidden_dropout_prob)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.attn_dropout_prob,
            activation=self.hidden_act,
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers
        )

        # Precompute causal mask
        causal_mask = self._generate_square_subsequent_mask(self.max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

        # Precompute item-to-feature mapping for fast lookup during forward
        # We store this as a buffer so it moves to GPU automatically
        for feature_name in self.selected_features:
            feat_tensor = torch.tensor(interactions.item_features[feature_name], dtype=torch.long)
            self.register_buffer(f"map_{feature_name}", feat_tensor)

        # Initialize weights
        self.apply(self._init_weights)

        # Loss function setup
        self.main_loss: nn.Module
        if self.neg_samples > 0:
            self.main_loss = BPRLoss()
        else:
            self.main_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

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

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        """Forward pass of the SASRecF model.

        Args:
            item_seq (Tensor): Padded sequences of item IDs [batch_size, max_seq_len].
            item_seq_len (Tensor): Actual lengths of sequences.

        Returns:
            Tensor: The session representation [batch_size, embedding_size].
        """
        # 1. Get Item ID Embeddings
        item_emb = self.item_embedding(item_seq) # [B, L, H]

        # 2. Get Feature Embeddings and Concatenate
        feature_embs = [item_emb]
        for feature_name in self.selected_features:
            # Map item IDs in sequence to their corresponding feature IDs
            # Note: item_seq contains indices 0..N-1, or self.n_items for padding
            # We handle padding index by ensuring the map has an entry for it
            feat_map = getattr(self, f"map_{feature_name}")
            
            # Create a version of the map that handles the padding index
            # (Padding item maps to the padding index of the feature embedding)
            padding_feat_idx = self.feature_embeddings[feature_name].padding_idx
            extended_map = torch.cat([feat_map, torch.tensor([padding_feat_idx], device=item_seq.device)])
            
            current_feat_seq = extended_map[item_seq]
            feature_embs.append(self.feature_embeddings[feature_name](current_feat_seq))

        # Concatenate item + all features along the last dimension
        input_concat = torch.cat(feature_embs, dim=-1) # [B, L, (1+F)*H]
        
        # Project back to hidden_size
        input_emb = self.concat_layer(input_concat) # [B, L, H]

        # 3. Add Position Embeddings
        seq_len = item_seq.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seq.device).unsqueeze(0)
        input_emb = input_emb + self.position_embedding(position_ids)

        # 4. Apply Normalization and Dropout
        input_emb = self.layernorm(input_emb)
        input_emb = self.emb_dropout(input_emb)

        # 5. Transformer Encoder Pass
        padding_mask = (item_seq == self.n_items)
        transformer_output = self.transformer_encoder(
            src=input_emb,
            mask=self.causal_mask[:seq_len, :seq_len],
            src_key_padding_mask=padding_mask
        )

        # 6. Gather the output of the last relevant item
        seq_output = self._gather_indexes(transformer_output, item_seq_len - 1)
        return seq_output

    def train_step(self, batch: Any, *args, **kwargs):
        """Execute a single training step."""
        if self.neg_samples > 0:
            item_seq, item_seq_len, pos_item, neg_item = batch
        else:
            item_seq, item_seq_len, pos_item = batch
            neg_item = None
        seq_output = self.forward(item_seq, item_seq_len)

        if self.neg_samples > 0:
            # Pairwise Loss (BPR)
            pos_items_emb = self.item_embedding(pos_item)
            neg_items_emb = self.item_embedding(neg_item)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output.unsqueeze(1) * neg_items_emb, dim=-1)
            main_loss = self.main_loss(pos_score, neg_score)
            
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq), pos_items_emb, neg_items_emb
            )
        else:
            # Pointwise Loss (Cross Entropy)
            test_item_emb = self.item_embedding.weight[:self.n_items]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            main_loss = self.main_loss(logits, pos_item)
            
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq), self.item_embedding(pos_item)
            )

        return main_loss + reg_loss

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
        """
        # Get sequence output embeddings
        seq_output = self.forward(user_seq, seq_len)

        if item_indices is None:
            # Full prediction on all items
            item_embeddings = self.item_embedding.weight[:-1, :]
            einsum_string = "be,ie->bi"
        else:
            # Prediction on a sampled set of items
            item_embeddings = self.item_embedding(item_indices)
            einsum_string = "be,bse->bs"

        predictions = torch.einsum(einsum_string, seq_output, item_embeddings)
        return predictions