# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn
# @Adapted for WarpRec framework

"""
DuoRec - Dual Contrastive Learning for Sequential Recommendation
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.
    Adapted with contrastive learning mechanisms.

Reference:
    https://github.com/kang205/SASRec

"""

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


@model_registry.register(name="DuoRec")
class DuoRec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of DuoRec algorithm - Dual contrastive learning for sequential recommendation.

    DuoRec combines self-attention mechanisms with unsupervised and supervised contrastive learning
    for improved sequential recommendation. It extends SASRec with dual contrastive objectives.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        embedding_size (int): The dimension of the item embeddings.
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
        ssl_type (str): Type of self-supervised learning ('us', 'su', 'us_x', 'none').
        ssl_lambda (float): Weight for unsupervised contrastive loss.
        ssl_lambda_sem (float): Weight for supervised contrastive loss.
        tau (float): Temperature parameter for contrastive learning.
        sim_type (str): Similarity metric ('dot' or 'cos').
        crop_eta (float): Ratio for cropping augmentation.
        mask_gamma (float): Ratio for masking augmentation.
        reorder_beta (float): Ratio for reordering augmentation.
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
    ssl_type: str = "none"
    ssl_lambda: float = 0.5
    ssl_lambda_sem: float = 0.5
    tau: float = 0.1
    sim_type: str = "dot"

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

        # Item and position embeddings
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)

        # Transformer encoder
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

        # Precompute position IDs to avoid recreation in forward pass
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long)
        self.register_buffer("position_ids", position_ids)

        # Initialize weights
        self.apply(self._init_weights)

        # Loss function selection
        self.main_loss: nn.Module
        if self.neg_samples > 0:
            self.main_loss = BPRLoss()
        else:
            self.main_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

        # Contrastive learning losses (same as what RecBole uses)
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.sem_aug_nce_fct = nn.CrossEntropyLoss()
        
        # Precompute mask for contrastive samples
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)

    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def mask_correlated_samples(self, batch_size: int) -> Tensor:
        """Create a mask for correlated samples in contrastive learning."""
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

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
        """Train step following RecBole's DuoRec implementation.
        
        Expects batch format:
        - BPR loss: (item_seq, item_seq_len, pos_item, neg_item)
        - CE loss: (item_seq, item_seq_len, pos_item)
        - With SSL: additionally sem_aug, sem_aug_lengths in kwargs
        """
        if self.neg_samples > 0:
            item_seq, item_seq_len, pos_item, neg_item = batch
        else:
            item_seq, item_seq_len, pos_item = batch
            neg_item = None

        seq_output = self.forward(item_seq, item_seq_len)

        # Calculate main loss (BPR or CrossEntropy)
        if self.neg_samples > 0:
            pos_items_emb = self.item_embedding(pos_item)
            neg_items_emb = self.item_embedding(neg_item)

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            # neg_items_emb shape: [batch_size, neg_samples, embedding_size]
            # After unsqueeze(1): seq_output is [batch_size, 1, embedding_size]
            # After multiplication and sum: [batch_size, neg_samples]
            neg_scores = torch.sum(seq_output.unsqueeze(1) * neg_items_emb, dim=-1)
            # Take the max (hardest negative) for BPR loss
            neg_score = neg_scores.max(dim=-1)[0]
            loss = self.main_loss(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.main_loss(logits, pos_item)

        # Skip SSL losses if ssl_type is 'none' - saves computation
        if self.ssl_type == "none":
            return loss

        # Unsupervised contrastive learning - ONLY if augmented data is provided
        if self.ssl_type in ["us", "un"]:
            # Only apply if augmented data is actually provided
            if "aug_item_seq" in kwargs and "aug_item_seq_len" in kwargs:
                aug_seq = kwargs["aug_item_seq"]
                aug_seq_len = kwargs["aug_item_seq_len"]
                aug_seq_output = self.forward(aug_seq, aug_seq_len)
                
                nce_logits, nce_labels = self.info_nce(
                    seq_output, aug_seq_output, 
                    temp=self.tau,
                    batch_size=item_seq_len.shape[0], 
                    sim=self.sim_type
                )
                loss += self.ssl_lambda * self.aug_nce_fct(nce_logits, nce_labels)

        # Supervised contrastive learning
        if self.ssl_type in ["su", "us"]:
            if "sem_aug" in kwargs and "sem_aug_lengths" in kwargs:
                sem_aug = kwargs["sem_aug"]
                sem_aug_lengths = kwargs["sem_aug_lengths"]
                sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths)

                sem_nce_logits, sem_nce_labels = self.info_nce(
                    seq_output, sem_aug_seq_output, 
                    temp=self.tau,
                    batch_size=item_seq_len.shape[0], 
                    sim=self.sim_type
                )
                loss += self.ssl_lambda_sem * self.sem_aug_nce_fct(sem_nce_logits, sem_nce_labels)

        # Special case for us_x: unsupervised and supervised contrastive learning together
        if self.ssl_type == "us_x":
            # Only process if we have both types of augmentations
            has_unsup = "aug_item_seq" in kwargs and "aug_item_seq_len" in kwargs
            has_sup = "sem_aug" in kwargs and "sem_aug_lengths" in kwargs
            
            if has_unsup and has_sup:
                aug_seq = kwargs["aug_item_seq"]
                aug_seq_len = kwargs["aug_item_seq_len"]
                aug_seq_output = self.forward(aug_seq, aug_seq_len)

                sem_aug = kwargs["sem_aug"]
                sem_aug_lengths = kwargs["sem_aug_lengths"]
                sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths)

                # Apply contrastive loss between unsupervised and supervised augmentations
                sem_nce_logits, sem_nce_labels = self.info_nce(
                    aug_seq_output, sem_aug_seq_output, 
                    temp=self.tau,
                    batch_size=item_seq_len.shape[0], 
                    sim=self.sim_type
                )
                loss += self.ssl_lambda_sem * self.sem_aug_nce_fct(sem_nce_logits, sem_nce_labels)

        return loss

    def get_attention_mask(self, item_seq: Tensor) -> Tensor:
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_bi_attention_mask(self, item_seq: Tensor) -> Tensor:
        """Generate bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        """Forward pass of the DuoRec model following RecBole's approach.

        Args:
            item_seq (Tensor): Padded sequences of item IDs [batch_size, max_seq_len].
            item_seq_len (Tensor): Actual lengths of sequences [batch_size,].

        Returns:
            Tensor: The embedding of the predicted item (last session state)
                    [batch_size, embedding_size].
        """
        seq_len = item_seq.size(1)
        device = item_seq.device
        
        # Use pre-registered position_ids buffer, sliced to actual sequence length
        position_ids = self.position_ids[:seq_len].unsqueeze(0).expand(item_seq.size(0), -1)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.layernorm(input_emb)
        input_emb = self.emb_dropout(input_emb)

        # Get padding mask once
        padding_mask = (item_seq == self.n_items)

        # Use cached causal mask if applicable, otherwise generate
        if hasattr(self, 'causal_mask') and self.causal_mask.shape[0] == seq_len:
            attention_mask = self.causal_mask.to(device)
        else:
            # Generate causal mask for this sequence length
            attention_mask = self.get_attention_mask(item_seq)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.squeeze(1)[0]

        transformer_output = self.transformer_encoder(
            src=input_emb,
            mask=attention_mask,
            src_key_padding_mask=padding_mask,
        )

        # Gather the output of the last relevant item in each sequence
        output = self._gather_indexes(transformer_output, item_seq_len - 1)
        return output

    def info_nce(
        self, z_i: Tensor, z_j: Tensor, temp: float, batch_size: int, sim: str = "dot"
    ) -> tuple:
        """
        Compute InfoNCE loss for contrastive learning following RecBole's implementation.
        
        Optimized to avoid creating full NxN similarity matrix for large batch sizes.

        Args:
            z_i (Tensor): First set of embeddings [batch_size, embedding_size].
            z_j (Tensor): Second set of embeddings [batch_size, embedding_size].
            temp (float): Temperature for similarity scaling.
            batch_size (int): Batch size.
            sim (str): Similarity metric ('dot' or 'cos').

        Returns:
            tuple: (logits, labels) for contrastive loss.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)  # [2*batch_size, embedding_size]
        
        # Compute similarities efficiently without creating full matrix
        if sim == "cos":
            # Normalize embeddings for cosine similarity
            z_norm = torch.nn.functional.normalize(z, dim=1)
            # Compute all pairwise similarities: [2N, 2N]
            sim_matrix = torch.mm(z_norm, z_norm.T) / temp
        elif sim == "dot":
            # Use lower precision computation if possible to save memory
            with torch.cuda.amp.autocast(enabled=z.is_cuda):
                sim_matrix = torch.mm(z, z.T) / temp
        else:
            raise ValueError(f"Unknown similarity metric: {sim}")
        
        # Extract positive pairs (diagonals) - these are cheap
        sim_i_j = torch.diag(sim_matrix, batch_size)
        sim_j_i = torch.diag(sim_matrix, -batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        # Extract negative samples efficiently using mask
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size).to(sim_matrix.device)
        else:
            mask = self.mask_default.to(sim_matrix.device)
        
        negative_samples = sim_matrix[mask].reshape(N, -1)
        
        labels = torch.zeros(N, dtype=torch.long, device=z.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        return logits, labels

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
        seq_output = self.forward(user_seq, seq_len)

        if item_indices is None:
            # Case 'full': prediction on all items
            item_embeddings = self.item_embedding.weight[:-1, :]
            einsum_string = "be,ie->bi"
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = self.item_embedding(item_indices)
            einsum_string = "be,bse->bs"

        predictions = torch.einsum(einsum_string, seq_output, item_embeddings)
        return predictions
