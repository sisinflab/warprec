# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class TransNet(nn.Module):
    """Transformer-based Encoder for CORE.
    
    This module implements the Representation-Consistent Encoder (RCE) logic
    by processing the sequence through a Transformer and outputting 
    attention weights (alpha) for each item in the session.
    """
    def __init__(self, params, max_seq_len, padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        
        hidden_size = params["embedding_size"]
        n_heads = params["n_heads"]
        inner_size = params["inner_size"]
        n_layers = params["n_layers"]
        dropout = params["hidden_dropout_prob"]
        
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=inner_size,
            dropout=dropout,
            activation=params["hidden_act"],
            layer_norm_eps=params["layer_norm_eps"],
            batch_first=True
        )
        
        self.trm_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers
        )

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=params["layer_norm_eps"])
        self.dropout = nn.Dropout(dropout)
        self.fn = nn.Linear(hidden_size, 1)

    def forward(self, item_seq, item_emb):
        # Create masks
        key_padding_mask = (item_seq == self.padding_idx)
        sz = item_seq.size(1)
        causal_mask = torch.triu(torch.ones(sz, sz, device=item_seq.device) * float('-inf'), diagonal=1)

        # Add position embeddings
        position_ids = torch.arange(sz, dtype=torch.long, device=item_seq.device).unsqueeze(0)
        position_embedding = self.position_embedding(position_ids)

        x = item_emb + position_embedding
        x = self.LayerNorm(x)
        x = self.dropout(x)

        # Transformer pass
        trm_output = self.trm_encoder(
            x, 
            mask=causal_mask, 
            src_key_padding_mask=key_padding_mask
        )

        # Calculate attention weights (alpha) for RCE
        alpha = self.fn(trm_output).to(torch.double)
        mask_for_alpha = (item_seq != self.padding_idx).unsqueeze(-1)
        alpha = torch.where(mask_for_alpha, alpha, -9e15)
        alpha = torch.softmax(alpha, dim=1, dtype=torch.float)
        
        return alpha


@model_registry.register(name="CORE")
class CORE(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of CORE algorithm from
    "CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space." in SIGIR 2022.

    CORE unifies the representation space for both encoding and decoding processes,
    using a Representation-Consistent Encoder (RCE) and Robust Distance Measuring (RDM).

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The dimension of the item embeddings.
        dnn_type (str): Type of encoder ('trm' for Transformer or 'ave' for Average).
        n_layers (int): Number of transformer layers.
        n_heads (int): Number of attention heads.
        inner_size (int): Inner size of the transformer feed-forward layer.
        hidden_dropout_prob (float): Dropout probability for hidden layers.
        attn_dropout_prob (float): Dropout probability for attention weights.
        hidden_act (str): Activation function for the transformer.
        layer_norm_eps (float): Epsilon for layer normalization.
        initializer_range (float): Range for weight initialization.
        sess_dropout (float): Dropout for the session embeddings.
        item_dropout (float): Dropout for item embeddings during training.
        temperature (float): Temperature scaling factor for RDM.
        max_seq_len (int): Maximum sequence length.
    """

    DATALOADER_TYPE = DataLoaderType.SEQUENTIAL_LOADER

    # Model hyperparameters (mapped from YAML)
    embedding_size: int
    dnn_type: str
    n_layers: int
    n_heads: int
    inner_size: int
    hidden_dropout_prob: float
    attn_dropout_prob: float
    hidden_act: str
    layer_norm_eps: float
    initializer_range: float
    sess_dropout: float
    item_dropout: float
    temperature: float
    max_seq_len: int
    batch_size: int
    reg_weight: float
    weight_decay: float
    epochs: int
    learning_rate: float
    neg_samples: int

    def __init__(self, params: dict, info: dict, *args, **kwargs):
        super().__init__(params, info, *args, **kwargs)

        # Padding index is set to the last index
        self.padding_idx = self.n_items 
        
        self.sess_dropout_layer = nn.Dropout(self.sess_dropout)
        self.item_dropout_layer = nn.Dropout(self.item_dropout)
        
        self.item_embedding = nn.Embedding(
            self.n_items + 1, 
            self.embedding_size, 
            padding_idx=self.padding_idx
        )

        # Initialize the chosen DNN encoder
        if self.dnn_type == "trm":
            self.net = TransNet(params, self.max_seq_len, self.padding_idx)
        else:
            self.net = self.ave_net

        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def ave_net(self, item_seq: Tensor, item_emb: Tensor) -> Tensor:
        """Simple average pooling encoder."""
        mask = (item_seq != self.padding_idx).to(torch.float)
        alpha = mask / mask.sum(dim=-1, keepdim=True)
        return alpha.unsqueeze(-1)

    def forward(self, item_seq: Tensor, item_seq_len: Tensor = None) -> Tensor:
        """Forward pass of the CORE model.

        Args:
            item_seq (Tensor): Padded sequences of item IDs [batch_size, max_seq_len].
            item_seq_len (Tensor): Actual lengths of sequences.

        Returns:
            Tensor: The session representation [batch_size, embedding_size].
        """
        # Get item embeddings
        x = self.item_embedding(item_seq)
        x = self.sess_dropout_layer(x)
        
        # Representation-Consistent Encoder (RCE):
        # Calculate weights alpha and perform weighted sum of item embeddings
        alpha = self.net(item_seq, x)
        seq_output = torch.sum(alpha * x, dim=1)
        
        # Normalize output for Robust Distance Measuring (RDM)
        return F.normalize(seq_output, dim=-1)
    
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
        """Execute a single training step.

        Args:
            batch (Any): The batch of data containing item sequences and targets.

        Returns:
            Tensor: The calculated loss value.
        """
        item_seq, item_seq_len, pos_item = batch[:3]

        # Generate session representation
        seq_output = self.forward(item_seq, item_seq_len)

        # Robust Distance Measuring (RDM):
        # Calculate cosine similarity between session and all items
        all_item_emb = self.item_embedding.weight[:self.n_items] 
        all_item_emb = self.item_dropout_layer(all_item_emb)
        all_item_emb = F.normalize(all_item_emb, dim=-1)

        # Logits are scaled by temperature
        logits = torch.matmul(seq_output, all_item_emb.transpose(0, 1)) / self.temperature
        return self.loss_fct(logits, pos_item)

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
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            user_seq (Optional[Tensor]): Padded sequences of item IDs for users.
            seq_len (Optional[Tensor]): Actual lengths of these sequences.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Get session representation
        seq_output = self.forward(user_seq, seq_len)

        if item_indices is None:
            # Predict scores for all items
            item_embeddings = self.item_embedding.weight[:-1, :]
            item_embeddings = F.normalize(item_embeddings, dim=-1)
            einsum_string = "be,ie->bi"
        else:
            # Predict scores for a specific subset of items
            item_embeddings = self.item_embedding(item_indices)
            item_embeddings = F.normalize(item_embeddings, dim=-1)
            einsum_string = "be,bse->bs"

        # Calculate similarity and apply temperature scaling
        predictions = torch.einsum(einsum_string, seq_output, item_embeddings)
        return predictions / self.temperature