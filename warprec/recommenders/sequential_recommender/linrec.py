# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_normal_, constant_

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.data.entities import Interactions, Sessions
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class LRULayer(nn.Module):
    """
    A Simplified Linear Recurrent Unit (LRU) block.
    Captures long-range dependencies with O(L) complexity.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Diagonal transition matrix (lambda) - parameterized in log space for stability
        # We want lambda to be between 0 and 1.
        self.lambda_log = nn.Parameter(torch.log(torch.uniform_(torch.empty(d_model), 0.7, 0.9)))
        self.gamma = nn.Parameter(torch.ones(d_model))
        
        self.input_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, L, D]
        b, l, d = x.size()
        u = self.input_proj(x)
        
        # Parallel prefix sum (linear recurrence)
        # h_t = lambda * h_{t-1} + gamma * u_t
        lambdas = torch.exp(self.lambda_log) # [D]
        
        # Expanding lambdas to compute the cumulative decay powers
        # For a true "2025" implementation, one might use the associative scan 
        # from the 'heinsen_sequence' or 'flash-linear-attention' libs.
        # Here we use a stable sequential-style simulation for the framework.
        h = torch.zeros(b, d, device=x.device)
        outputs = []
        for t in range(l):
            h = lambdas * h + self.gamma * u[:, t, :]
            outputs.append(h.unsqueeze(1))
        
        y = torch.cat(outputs, dim=1)
        return self.dropout(self.output_proj(y)) + x # Residual connection


@model_registry.register(name="LinRec")
class LinRec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of LinRec (Linear Recurrent Sequential Recommendation).
    
    LinRec replaces heavy Attention or Gated RNNs with Linear Recurrent Units (LRU).
    It is significantly faster for long sequences and avoids the 'forgetting' 
    problems of traditional LSTMs.

    Attributes:
        embedding_size (int): Item embedding dimensions.
        hidden_size (int): Hidden size of the LRU layers.
        n_layers (int): Number of stacked LRU blocks.
        dropout_prob (float): Dropout probability.
        reg_weight (float): L2 regularization weight.
    """

    DATALOADER_TYPE = DataLoaderType.SEQUENTIAL_LOADER

    embedding_size: int
    hidden_size: int
    n_layers: int
    dropout_prob: float
    reg_weight: float
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

        # 1. Embeddings
        self.item_embedding = nn.Embedding(
            self.n_items + 1,
            self.embedding_size,
            padding_idx=self.n_items,
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        
        # 2. Linear Recurrent Encoder
        # stack of LRU layers
        self.encoder = nn.ModuleList([
            LRULayer(self.embedding_size) for _ in range(self.n_layers)
        ])
        
        # 3. Final Layer Norm and Projection
        self.layer_norm = nn.LayerNorm(self.embedding_size)
        self.out_proj = nn.Linear(self.embedding_size, self.embedding_size)

        self.apply(self._init_weights)

        # Loss logic
        if self.neg_samples > 0:
            self.main_loss = BPRLoss()
        else:
            self.main_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def get_dataloader(self, interactions, sessions, low_memory=False, **kwargs):
        return sessions.get_sequential_dataloader(
            max_seq_len=self.max_seq_len,
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
            low_memory=low_memory,
        )

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        # Mask for padding
        item_seq_emb = self.item_embedding(item_seq)
        x = self.emb_dropout(item_seq_emb)

        # Pass through stacked Linear Recurrent Units
        for layer in self.encoder:
            x = layer(x)

        x = self.layer_norm(x)
        
        # Gather the representation at the last non-padded index
        seq_output = self._gather_indexes(x, item_seq_len - 1)
        
        # Final non-linear projection for head
        seq_output = self.out_proj(seq_output)
        
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
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        user_seq: Optional[Tensor] = None,
        seq_len: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        seq_output = self.forward(user_seq, seq_len)

        if item_indices is None:
            item_embeddings = self.item_embedding.weight[:-1, :]
            einsum_string = "be,ie->bi"
        else:
            item_embeddings = self.item_embedding(item_indices)
            einsum_string = "be,bse->bs"

        return torch.einsum(einsum_string, seq_output, item_embeddings)