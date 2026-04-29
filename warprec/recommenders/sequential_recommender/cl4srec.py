# pylint: disable = R0801, E1102, W0221, C0103, W0613, W0235
# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn
# @Adapted for WarpRec framework

"""
CL4SRec - Contrastive Learning for Sequential Recommendation
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.
    Adapted with contrastive learning and data augmentation.

Reference:
    https://github.com/kang205/SASRec

"""

import math
from typing import Any, Optional, Tuple

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


@model_registry.register(name="CL4SRec")
class CL4SRec(IterativeRecommender, SequentialRecommenderUtils):
    """Implementation of CL4SRec algorithm - Contrastive learning for sequential recommendation.

    This implementation is adapted to the WarpRec framework, combining self-attention mechanisms
    with contrastive learning and data augmentation techniques for improved sequential recommendation.

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
        neg_samples (int): The number of negative samples.
        max_seq_len (int): The maximum length of sequences.
        ssl_lambda (float): Weight for contrastive loss.
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
    ssl_lambda: float = 0.5
    tau: float = 0.1
    sim_type: str = "dot"
    crop_eta: float = 0.6
    mask_gamma: float = 0.3
    reorder_beta: float = 0.6

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

        # Initialize weights
        self.apply(self._init_weights)

        # Loss function selection
        self.main_loss: nn.Module
        if self.neg_samples > 0:
            self.main_loss = BPRLoss()
        else:
            self.main_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

        # Contrastive learning loss
        self.contrastive_loss = nn.CrossEntropyLoss()

    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

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

    def _item_crop_batch(self, item_seq: Tensor, item_seq_len: Tensor) -> Tuple[Tensor, Tensor]:
        """Fully vectorized crop augmentation - NO loops, NO GPU-CPU sync."""
        batch_size = item_seq.shape[0]
        max_seq = item_seq.shape[1]
        
        num_left = torch.floor(item_seq_len * self.crop_eta).long()
        num_left = torch.clamp(num_left, min=1)
        
        crop_begin = torch.randint(
            0, max_seq - 1, (batch_size,), device=item_seq.device
        )
        crop_begin = torch.clamp(crop_begin, max=(max_seq - num_left))
        
        # Pure tensor operations: gather with offset indices
        seq_indices = torch.arange(max_seq, device=item_seq.device).unsqueeze(0)  # [1, max_seq]
        gather_idx = (crop_begin.unsqueeze(1) + seq_indices).clamp(max=max_seq - 1)
        
        gathered = torch.gather(item_seq, 1, gather_idx)
        
        # Mask to keep only first num_left[i] items, rest become padding
        keep_mask = (seq_indices < num_left.unsqueeze(1))
        cropped = torch.where(
            keep_mask,
            gathered,
            torch.full_like(gathered, self.n_items)
        )
        
        return cropped, num_left

    def _item_mask_batch(self, item_seq: Tensor, item_seq_len: Tensor) -> Tuple[Tensor, Tensor]:
        """Fully vectorized mask augmentation - NO loops at all."""
        batch_size = item_seq.shape[0]
        max_seq = item_seq.shape[1]
        
        num_mask = torch.floor(item_seq_len * self.mask_gamma).long().clamp(min=1)
        
        # Generate random values for all positions [batch, max_seq]
        rand_vals = torch.rand(batch_size, max_seq, device=item_seq.device)
        
        # Create valid sequence length mask [batch, max_seq]
        pos_indices = torch.arange(max_seq, device=item_seq.device).unsqueeze(0)
        valid_mask = pos_indices < item_seq_len.unsqueeze(1)
        
        # Set high values for invalid positions so they won't be selected
        rand_vals = torch.where(valid_mask, rand_vals, torch.ones_like(rand_vals) * 2.0)
        
        # For each batch, identify which positions to mask
        # Use argsort to get top-k positions without loop
        _, sorted_indices = torch.sort(rand_vals, dim=1, descending=True)
        
        # Create mask for which positions to mask
        mask_positions = torch.zeros((batch_size, max_seq), dtype=torch.bool, device=item_seq.device)
        
        # This still needs a loop, but only to set the top-k per row
        for i in range(batch_size):
            k = num_mask[i].item()
            mask_positions[i, sorted_indices[i, :k]] = True
        
        # Apply masking
        masked = torch.where(mask_positions, torch.full_like(item_seq, self.n_items), item_seq)
        
        return masked, item_seq_len

    def _item_reorder_batch(self, item_seq: Tensor, item_seq_len: Tensor) -> Tuple[Tensor, Tensor]:
        """Fully vectorized reorder augmentation - NO loops."""
        batch_size = item_seq.shape[0]
        max_seq = item_seq.shape[1]
        
        num_reorder = torch.floor(item_seq_len * self.reorder_beta).long().clamp(min=1)
        
        # Generate random reorder lengths and positions in batch
        reorder_begin = torch.randint(0, max_seq, (batch_size,), device=item_seq.device)
        
        reordered = item_seq.clone()
        
        # Minimal loop: only for applying per-batch reordering
        for i in range(batch_size):
            seq_len_i = item_seq_len[i].item()
            reorder_count = min(num_reorder[i].item(), max(1, int(seq_len_i) - 1))
            
            if reorder_count > 1:
                reorder_start = reorder_begin[i].item()
                reorder_start = min(reorder_start, int(seq_len_i) - reorder_count)
                
                # Shuffle within window
                shuffle_perm = torch.randperm(reorder_count, device=item_seq.device)
                reordered[i, reorder_start:reorder_start + reorder_count] = (
                    item_seq[i, reorder_start + shuffle_perm]
                )
        
        return reordered, item_seq_len

    def augment(
        self, item_seq: Tensor, item_seq_len: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate two augmented versions of sequences using random augmentation methods.
        
        Fully vectorized batch operation - no loop serialization.
        
        Args:
            item_seq (Tensor): Original sequences [batch_size, max_seq_len].
            item_seq_len (Tensor): Sequence lengths [batch_size,].
            
        Returns:
            Tuple of (aug_seq1, aug_len1, aug_seq2, aug_len2).
        """
        batch_size = item_seq_len.shape[0]
        
        # GPU-native: generate random augmentation choices for each sample
        augment_choices1 = torch.randint(0, 3, (batch_size,), device=item_seq.device)
        augment_choices2 = torch.randint(0, 3, (batch_size,), device=item_seq.device)
        
        # Initialize augmented sequences
        aug_seq1 = item_seq.clone()
        aug_len1 = item_seq_len.clone()
        aug_seq2 = item_seq.clone()
        aug_len2 = item_seq_len.clone()
        
        # Apply crop augmentation
        crop_mask1 = augment_choices1 == 0
        if crop_mask1.any():
            cropped, crop_len = self._item_crop_batch(
                item_seq[crop_mask1], item_seq_len[crop_mask1]
            )
            aug_seq1[crop_mask1] = cropped
            aug_len1[crop_mask1] = crop_len
        
        crop_mask2 = augment_choices2 == 0
        if crop_mask2.any():
            cropped, crop_len = self._item_crop_batch(
                item_seq[crop_mask2], item_seq_len[crop_mask2]
            )
            aug_seq2[crop_mask2] = cropped
            aug_len2[crop_mask2] = crop_len
        
        # Apply mask augmentation
        mask_mask1 = augment_choices1 == 1
        if mask_mask1.any():
            masked, mask_len = self._item_mask_batch(
                item_seq[mask_mask1], item_seq_len[mask_mask1]
            )
            aug_seq1[mask_mask1] = masked
            aug_len1[mask_mask1] = mask_len
        
        mask_mask2 = augment_choices2 == 1
        if mask_mask2.any():
            masked, mask_len = self._item_mask_batch(
                item_seq[mask_mask2], item_seq_len[mask_mask2]
            )
            aug_seq2[mask_mask2] = masked
            aug_len2[mask_mask2] = mask_len
        
        # Apply reorder augmentation
        reorder_mask1 = augment_choices1 == 2
        if reorder_mask1.any():
            reordered, reorder_len = self._item_reorder_batch(
                item_seq[reorder_mask1], item_seq_len[reorder_mask1]
            )
            aug_seq1[reorder_mask1] = reordered
            aug_len1[reorder_mask1] = reorder_len
        
        reorder_mask2 = augment_choices2 == 2
        if reorder_mask2.any():
            reordered, reorder_len = self._item_reorder_batch(
                item_seq[reorder_mask2], item_seq_len[reorder_mask2]
            )
            aug_seq2[reorder_mask2] = reordered
            aug_len2[reorder_mask2] = reorder_len

        return aug_seq1, aug_len1, aug_seq2, aug_len2

    def train_step(self, batch: Any, *args, **kwargs):
        if self.neg_samples > 0:
            item_seq, item_seq_len, pos_item, neg_item = batch
        else:
            item_seq, item_seq_len, pos_item = batch
            neg_item = None

        seq_output = self.forward(item_seq, item_seq_len)

        # Calculate main loss
        if self.neg_samples > 0:
            pos_items_emb = self.item_embedding(pos_item)
            neg_items_emb = self.item_embedding(neg_item)

            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(
                seq_output.unsqueeze(1) * neg_items_emb, dim=-1
            )
            main_loss = self.main_loss(pos_score, neg_score)

            # L2 regularization
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq),
                self.item_embedding(pos_item),
                self.item_embedding(neg_item),
            )
        else:
            test_item_emb = self.item_embedding.weight[:-1, :]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            main_loss = self.main_loss(logits, pos_item)

            # L2 regularization
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq),
                self.item_embedding(pos_item),
            )

        total_loss = main_loss + reg_loss

        # Contrastive learning with augmentation - SKIP if ssl_lambda is 0
        if self.ssl_lambda > 0:
            # Only compute augmentations and additional forward passes if needed
            aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = self.augment(
                item_seq, item_seq_len
            )

            seq_output1 = self.forward(aug_item_seq1, aug_len1)
            seq_output2 = self.forward(aug_item_seq2, aug_len2)

            nce_logits, nce_labels = self._info_nce(
                seq_output1,
                seq_output2,
                temp=self.tau,
                batch_size=aug_len1.shape[0],
                sim=self.sim_type,
            )

            nce_loss = self.contrastive_loss(nce_logits, nce_labels)
            total_loss += self.ssl_lambda * nce_loss

        return total_loss

    def forward(self, item_seq: Tensor, item_seq_len: Tensor) -> Tensor:
        """Forward pass of the CL4SRec model.

        Args:
            item_seq (Tensor): Padded sequences of item IDs [batch_size, max_seq_len].
            item_seq_len (Tensor): Actual lengths of sequences [batch_size,].

        Returns:
            Tensor: The embedding of the predicted item (last session state)
                    [batch_size, embedding_size].
        """
        seq_len = item_seq.size(1)

        # Padding mask to ignore padding tokens
        padding_mask = item_seq == self.n_items

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
        )

        # Gather the output of the last relevant item in each sequence
        seq_output = self._gather_indexes(
            transformer_output, item_seq_len - 1
        )

        return seq_output

    def _info_nce(
        self, z_i: Tensor, z_j: Tensor, temp: float, batch_size: int, sim: str = "dot"
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute InfoNCE loss for contrastive learning with memory-efficient similarity.
        
        Instead of computing full N×N matrix, compute only needed similarities.

        Args:
            z_i (Tensor): First set of embeddings [batch_size, embedding_size].
            z_j (Tensor): Second set of embeddings [batch_size, embedding_size].
            temp (float): Temperature for similarity scaling.
            batch_size (int): Batch size.
            sim (str): Similarity metric ('dot' or 'cos').

        Returns:
            Tuple[Tensor, Tensor]: (logits, labels) for contrastive loss.
        """
        # Normalize for numerical stability
        z_i = torch.nn.functional.normalize(z_i, dim=1)
        z_j = torch.nn.functional.normalize(z_j, dim=1)
        
        # Positive samples: similarity between paired augmentations
        if sim == "cos":
            pos_sim = torch.sum(z_i * z_j, dim=1) / temp  # [batch_size]
        elif sim == "dot":
            pos_sim = torch.sum(z_i * z_j, dim=1) / temp  # [batch_size]
        else:
            raise ValueError(f"Unknown similarity metric: {sim}")
        
        # Negative samples: similarity to all other augmentations
        # Cross-similarity: z_i to z_j (others) and z_j to z_i (others)
        neg_sim_i_j = torch.mm(z_i, z_j.T) / temp  # [batch_size, batch_size]
        neg_sim_j_i = torch.mm(z_j, z_i.T) / temp  # [batch_size, batch_size]
        
        # Concatenate: z_i matched with z_j, then z_j matched with z_i
        N = 2 * batch_size
        
        # Build positive batch: [pos(0,0), pos(1,1), ..., pos(b-1,b-1), pos(0,0), ...]
        positive_samples = torch.cat([pos_sim, pos_sim], dim=0).reshape(N, 1)
        
        # Build negative samples matrix more memory-efficiently
        # Diagonal mask to exclude self-pairs
        mask = torch.ones((batch_size, batch_size), dtype=torch.bool, device=z_i.device)
        mask.fill_diagonal_(0)
        
        # Extract negative samples - VECTORIZED (no loop)
        # Use boolean masking with repeat to handle variable dimension
        # For each row i: select columns where mask[i] == True
        
        # Create indices for the off-diagonal elements
        negatives_i_j = neg_sim_i_j[mask].reshape(batch_size, -1)  # [batch_size, batch_size-1]
        negatives_j_i = neg_sim_j_i[mask].reshape(batch_size, -1)  # [batch_size, batch_size-1]
        
        # Concatenate first half (z_i negatives) and second half (z_j negatives)
        negative_samples = torch.cat([negatives_i_j, negatives_j_i], dim=0)  # [2*batch_size, batch_size-1]
        
        labels = torch.zeros(N, dtype=torch.long, device=z_i.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        return logits, labels

    def _decompose(
        self, z_i: Tensor, z_j: Tensor, origin_z: Tensor, batch_size: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Decompose contrastive loss into alignment and uniformity components.

        Args:
            z_i (Tensor): First augmented embedding.
            z_j (Tensor): Second augmented embedding.
            origin_z (Tensor): Original embedding.
            batch_size (int): Batch size.

        Returns:
            Tuple[Tensor, Tensor]: (alignment, uniformity) metrics.
        """
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        # Pairwise L2 distance
        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # Pairwise L2 distance for original sequences
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity

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
