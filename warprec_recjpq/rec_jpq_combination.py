import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .centroid_assignment_strategies.centroid_strategy import CentroidAssignmentStragety
from .centroid_assignment_strategies.svd_strategy import SVDAssignmentStrategy


def get_codes_strategy(
    codes_strategy: str, item_code_bytes: int, num_items: int
) -> CentroidAssignmentStragety:
    if codes_strategy == "svd":
        return SVDAssignmentStrategy(item_code_bytes, num_items)
    raise ValueError(
        f"Unknown strategy: {codes_strategy}. Only 'svd' is supported in this WarpRec port."
    )


class ItemCodeLayer(nn.Module):
    """
    Enhanced ItemCodeLayer with PPS/SubID bonus support and optimized scoring methods.
    Combines logic from warprec and reference implementations.
    """
    
    def __init__(
        self,
        embedding_size: int,
        pq_m: int,
        num_items: int,
        sequence_length: int,
        codes_strategy: str,
        log_softmax: bool = False,
        exp: bool = False,
    ):
        super().__init__()
        if embedding_size % pq_m != 0:
            raise ValueError(
                f"embedding_size ({embedding_size}) must be divisible by pq_m ({pq_m})."
            )

        self.sub_embedding_size = embedding_size // pq_m
        self.item_code_bytes = pq_m
        self.sequence_length = sequence_length
        self.num_items = num_items
        self.log_softmax = log_softmax
        self.exp = exp

        if log_softmax and exp:
            raise ValueError("log_softmax and exp cannot be used together")

        self.vals_per_dim = (
            256 if codes_strategy != "qr" else math.ceil(math.sqrt(num_items))
        )
        self.base_type = torch.uint8 if codes_strategy != "qr" else torch.int32

        # Item codes buffer
        self.register_buffer(
            "item_codes",
            torch.zeros((num_items + 2, self.item_code_bytes), dtype=self.base_type),
        )
        
        # Codes count buffer for tracking frequency
        self.register_buffer(
            "codes_count",
            torch.ones((self.item_code_bytes, self.vals_per_dim), dtype=torch.float32),
        )
        
        # Bonus parameters for PPS/SubID
        self.register_buffer("alpha", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("beta", torch.tensor(0.0, dtype=torch.float32))

        # Trainable centroids
        self.centroids = nn.Parameter(
            torch.empty(
                (self.item_code_bytes, self.vals_per_dim, self.sub_embedding_size)
            )
        )
        nn.init.uniform_(self.centroids, -0.05, 0.05)

        # Initialize code assignment strategy
        if isinstance(codes_strategy, str):
            self.item_codes_strategy = get_codes_strategy(
                codes_strategy, self.item_code_bytes, num_items
            )
        elif isinstance(codes_strategy, CentroidAssignmentStragety):
            self.item_codes_strategy = codes_strategy
        else:
            raise TypeError("Invalid code assignment strategy")

    def pass_mapping(self, item_mappings):
        """Pass item mappings to the code assignment strategy."""
        self.item_codes_strategy.pass_item_mappings(item_mappings)

    def assign_codes(self, train_users):
        """Assign quantized codes to items based on training data."""
        codes = self.item_codes_strategy.assign(train_users)
        self.item_codes.copy_(
            torch.tensor(codes, dtype=self.base_type, device=self.item_codes.device)
        )

        # Update codes count
        counter = np.zeros((self.item_code_bytes, self.vals_per_dim), dtype=np.float32)
        npcodes = self.item_codes.cpu().numpy()
        
        for i in range(npcodes.shape[0]):
            for j in range(self.item_code_bytes):
                val = int(npcodes[i, j])
                if val < self.vals_per_dim:
                    counter[j, val] += 1.0

        self.codes_count.copy_(
            torch.tensor(
                counter, dtype=torch.float32, device=self.codes_count.device
            )
        )

    def map_subid_to_items(self) -> dict:
        """Map each (byte_idx, code_value) to list of items having that code."""
        codes_np = self.item_codes.cpu().numpy()
        mapping = {}
        for item_index in range(codes_np.shape[0] - 2):
            code = codes_np[item_index]
            for byte_idx, subid in enumerate(code):
                key = (byte_idx, int(subid))
                mapping.setdefault(key, []).append(item_index)
        return mapping

    def forward(self, input_ids: torch.Tensor, batch_size: int = None):
        """
        Get embeddings for input item IDs using quantized codes.
        
        Args:
            input_ids: (batch, seq_len) or any shape
            batch_size: optional batch size
            
        Returns:
            embeddings: same shape as input_ids + (embedding_size,)
        """
        input_codes = self.item_codes[input_ids.long()].long().detach()
        sub_embeddings = []
        
        for dim_idx in range(self.item_code_bytes):
            sub_emb = F.embedding(input_codes[..., dim_idx], self.centroids[dim_idx])
            sub_embeddings.append(sub_emb)
        
        return torch.cat(sub_embeddings, dim=-1)

    def score_all_centroids(self, seq_emb: torch.Tensor, batch_size: int = None):
        """
        Compute scores for all centroids.
        
        Args:
            seq_emb: (batch, seq_len, embedding_size)
            batch_size: optional batch size
            
        Returns:
            scores: (batch, seq_len, item_code_bytes, vals_per_dim)
        """
        if batch_size is None:
            batch_size = seq_emb.shape[0]
        
        seq_len = seq_emb.shape[1]
        sub_emb = seq_emb.view(batch_size, seq_len, self.item_code_bytes, self.sub_embedding_size)
        scores = torch.einsum("bsie,ine->bsin", sub_emb, self.centroids)
        
        if self.log_softmax:
            return F.log_softmax(scores, dim=-1)
        if self.exp:
            return torch.exp(scores)
        return scores

    def get_target_codes(self, target_ids: torch.Tensor):
        """Get quantized codes for target item IDs."""
        target_ids = F.relu(target_ids).long()
        codes = self.item_codes[target_ids]
        return codes.long()

    def score_sequence_items(self, seq_emb: torch.Tensor, target_ids: torch.Tensor, batch_size: int = None):
        """
        Score specific items for each sequence position (optimized with advanced indexing).
        
        Args:
            seq_emb: (batch_size, seq_len, embedding_size)
            target_ids: (batch_size, seq_len, num_targets) - e.g. [positive, neg1, neg2, ...]
            batch_size: batch size
            
        Returns:
            logits: (batch_size, seq_len, num_targets)
        """
        if batch_size is None:
            batch_size = seq_emb.shape[0]
        
        seq_len = seq_emb.shape[1]
        num_targets = target_ids.shape[2]
        
        # Reshape: (batch, seq_len, item_code_bytes, sub_embedding_size)
        sub_emb = seq_emb.view(batch_size, seq_len, self.item_code_bytes, self.sub_embedding_size)
        
        # Compute scores for all byte positions and all centroid values
        # Result: (batch, seq_len, item_code_bytes, vals_per_dim)
        scores = torch.einsum("bsie,ine->bsin", sub_emb, self.centroids)
        
        if self.log_softmax:
            scores = F.log_softmax(scores, dim=-1)
        elif self.exp:
            scores = torch.exp(scores)
        
        # Get codes for target items: (batch, seq_len, num_targets, item_code_bytes)
        target_codes = self.item_codes[F.relu(target_ids).long()].long()
        
        # Use advanced indexing to avoid OOM (no expand!)
        batch_idx = torch.arange(batch_size, device=scores.device).view(batch_size, 1, 1, 1)
        seq_idx = torch.arange(seq_len, device=scores.device).view(1, seq_len, 1, 1)
        byte_idx = torch.arange(self.item_code_bytes, device=scores.device).view(1, 1, 1, self.item_code_bytes)
        
        # Expand to match target_codes shape
        batch_idx = batch_idx.expand(batch_size, seq_len, num_targets, self.item_code_bytes)
        seq_idx = seq_idx.expand(batch_size, seq_len, num_targets, self.item_code_bytes)
        byte_idx = byte_idx.expand(batch_size, seq_len, num_targets, self.item_code_bytes)
        
        # Gather using advanced indexing: scores[batch, seq, byte, code]
        sub_scores = scores[batch_idx, seq_idx, byte_idx, target_codes]
        # sub_scores: (batch, seq_len, num_targets, item_code_bytes)
        
        # Sum over bytes: (batch, seq_len, num_targets)
        logits = torch.sum(sub_scores, dim=3)
        
        return logits

    def score_sequence_items_with_bonus_user(
        self, 
        seq_emb: torch.Tensor, 
        target_ids: torch.Tensor, 
        user_sequence_ids: torch.Tensor,
        batch_size: int = None
    ):
        """
        Score specific items for each sequence position with user bonuses.
        
        Args:
            seq_emb: (batch_size, seq_len, embedding_size)
            target_ids: (batch_size, seq_len, num_targets)
            user_sequence_ids: (batch_size, seq_len) - user history for bonus calculation
            batch_size: batch size
            
        Returns:
            logits: (batch_size, seq_len, num_targets)
        """
        if batch_size is None:
            batch_size = seq_emb.shape[0]
        
        seq_len = seq_emb.shape[1]
        num_targets = target_ids.shape[2]
        
        # Base PQ scores
        base_logits = self.score_sequence_items(seq_emb, target_ids, batch_size)
        
        # Calculate bonuses using user sequences - once per user, not per position!
        # user_sequence_ids: (batch_size, seq_len)
        # Calculate bonuses: (batch_size, num_items)
        pps_std, subid_std = self._calculate_bonuses(user_sequence_ids)
        
        # Expand to (batch, seq_len, num_items) for each position
        pps_std = pps_std.unsqueeze(1).expand(-1, seq_len, -1)
        subid_std = subid_std.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Gather bonuses for target items
        # target_ids: (batch, seq_len, num_targets)
        # Clamp to valid item range [0, num_items-1]
        num_items = pps_std.shape[-1]
        target_ids_clamped = torch.clamp(target_ids, 0, num_items - 1).long()
        
        # Use gather instead of advanced indexing to avoid CUDA issues
        # pps_std, subid_std: (batch, seq_len, num_items)
        # target_ids_clamped: (batch, seq_len, num_targets)
        pps_target = torch.gather(pps_std, 2, target_ids_clamped)
        subid_target = torch.gather(subid_std, 2, target_ids_clamped)
        
        # Combine: gamma * base + alpha * pps + beta * subid
        gamma = 1.0 - self.alpha - self.beta
        final_logits = gamma * base_logits + self.alpha * pps_target + self.beta * subid_target
        
        return final_logits

    def score_emb_items(self, seq_emb: torch.Tensor, target_ids: torch.Tensor, batch_size: int = None):
        """
        Score specific items for single embeddings (used for validation).
        
        Args:
            seq_emb: (batch_size, embedding_size)
            target_ids: (batch_size, num_targets) or (batch_size,)
            
        Returns:
            logits: (batch_size, num_targets) or (batch_size,)
        """
        if batch_size is None:
            batch_size = seq_emb.shape[0]
        
        sub_emb = seq_emb.view(batch_size, 1, self.item_code_bytes, self.sub_embedding_size)
        scores = torch.einsum("bsie,ine->bsin", sub_emb, self.centroids)
        
        if self.log_softmax:
            scores = F.log_softmax(scores, dim=-1)
        elif self.exp:
            scores = torch.exp(scores)
        
        target_codes = self.item_codes[F.relu(target_ids).long()].long()
        if target_codes.dim() != 3:
            target_codes = target_codes.unsqueeze(1)
        
        sub_scores = scores.gather(-1, target_codes.unsqueeze(-1))
        return torch.sum(sub_scores.squeeze(-1), dim=-1).squeeze(1)

    def score_all_items(self, seq_emb: torch.Tensor):
        """
        Score all items for given embeddings.
        
        Args:
            seq_emb: (batch_size, embedding_size)
            
        Returns:
            logits: (batch_size, num_items)
        """
        batch_size = seq_emb.shape[0]
        sub_emb = seq_emb.view(batch_size, self.item_code_bytes, self.sub_embedding_size)
        scores = torch.einsum("bie,ine->bin", sub_emb, self.centroids)

        if self.log_softmax:
            scores = F.log_softmax(scores, dim=-1)
        elif self.exp:
            scores = torch.exp(scores)

        target_codes = self.item_codes[: self.num_items].long()
        final_scores = torch.zeros((batch_size, self.num_items), device=seq_emb.device)
        
        for dim_idx in range(self.item_code_bytes):
            byte_scores = scores[:, dim_idx, :]
            item_indices = target_codes[:, dim_idx]
            final_scores += byte_scores.index_select(-1, item_indices)
        
        return final_scores

    def score_masked_tokens(self, masked_embeddings: torch.Tensor):
        """
        Score all items for masked token embeddings (optimized for BERT-style training).
        
        Args:
            masked_embeddings: (num_masked, embedding_size)
            
        Returns:
            logits: (num_masked, num_items)
        """
        num_masked = masked_embeddings.shape[0]
        sub_emb = masked_embeddings.view(
            num_masked, self.item_code_bytes, self.sub_embedding_size
        )
        scores = torch.einsum("mie,ine->min", sub_emb, self.centroids)

        if self.log_softmax:
            scores = F.log_softmax(scores, dim=-1)
        elif self.exp:
            scores = torch.exp(scores)

        target_codes = self.item_codes[: self.num_items].long()
        final_scores = torch.zeros(
            (num_masked, self.num_items), device=masked_embeddings.device
        )
        
        for dim_idx in range(self.item_code_bytes):
            byte_scores = scores[:, dim_idx, :]
            item_indices = target_codes[:, dim_idx]
            final_scores.add_(byte_scores.index_select(-1, item_indices))
        
        return final_scores

    def score_sequence_all_items(self, seq_emb: torch.Tensor, batch_size: int = None):
        """
        Score all items for each position in sequence.
        
        Args:
            seq_emb: (batch_size, seq_len, embedding_size)
            batch_size: batch size
            
        Returns:
            logits: (batch_size, seq_len, num_items)
        """
        if batch_size is None:
            batch_size = seq_emb.shape[0]
        
        seq_len = seq_emb.shape[1]
        
        # Reshape: (batch, seq_len, item_code_bytes, sub_embedding_size)
        sub_emb = seq_emb.view(batch_size, seq_len, self.item_code_bytes, self.sub_embedding_size)
        
        # Compute scores: (batch, seq_len, item_code_bytes, vals_per_dim)
        scores = torch.einsum("bsie,ine->bsin", sub_emb, self.centroids)
        
        if self.log_softmax:
            scores = F.log_softmax(scores, dim=-1)
        elif self.exp:
            scores = torch.exp(scores)
        
        # Get codes for all items
        target_codes = self.item_codes[:self.num_items].long()
        
        # Score all items for each position
        final_scores = torch.zeros(
            (batch_size, seq_len, self.num_items), device=seq_emb.device
        )
        
        for dim_idx in range(self.item_code_bytes):
            byte_scores = scores[:, :, dim_idx, :]  # (batch, seq_len, vals_per_dim)
            item_indices = target_codes[:, dim_idx]  # (num_items,)
            gathered = byte_scores.index_select(-1, item_indices)  # (batch, seq_len, num_items)
            final_scores += gathered
        
        return final_scores

    def _calculate_bonuses(self, user_history_ids: torch.Tensor):
        """
        Calculate PPS (Popularity) and SubID (Code Frequency) bonuses.
        
        Args:
            user_history_ids: (batch, seq_len) or (num_masked,)
            
        Returns:
            pps_std: (batch, num_items) - standardized popularity bonus
            subid_std: (batch, num_items) - standardized code frequency bonus
        """
        eps = 1e-6
        device = user_history_ids.device
        
        # Handle both 1D and 2D inputs
        if user_history_ids.dim() == 1:
            user_history_ids = user_history_ids.unsqueeze(1)
        
        batch_size = user_history_ids.shape[0]
        
        # Mask for valid items
        valid_mask = (user_history_ids >= 0) & (user_history_ids < self.num_items)
        safe_history = torch.where(
            valid_mask, 
            user_history_ids.long(), 
            torch.zeros_like(user_history_ids, dtype=torch.long)
        )

        # 1. PPS Bonus (Item Frequency)
        pps_counts = torch.zeros((batch_size, self.num_items), device=device)
        pps_counts.scatter_add_(1, safe_history, valid_mask.float())
        log_pps = torch.log(pps_counts + eps)

        # 2. SubID Bonus (Code Frequency)
        log_subid = torch.zeros((batch_size, self.num_items), device=device)
        all_item_codes = self.item_codes[:self.num_items].long()  # (num_items, pq_m)
        user_codes = self.item_codes[safe_history].long()  # (batch, seq_len, pq_m)

        for m in range(self.item_code_bytes):
            hist = torch.zeros((batch_size, self.vals_per_dim), device=device)
            hist.scatter_add_(1, user_codes[:, :, m], valid_mask.float())
            
            # Project histogram onto catalog items
            log_subid += torch.log(hist[:, all_item_codes[:, m]] + eps)

        # Standardization (Z-Score)
        def standardize(x):
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
            return (x - mean) / (std + eps)
        
        return standardize(log_pps), standardize(log_subid)

    def score_all_items_with_bonus_user(self, seq_emb: torch.Tensor, user_sequence_ids: torch.Tensor):
        """
        Score all items with personalized PPS/SubID bonus (for inference).
        
        Args:
            seq_emb: (batch_size, embedding_size)
            user_sequence_ids: (batch_size, seq_len) - user's historical item IDs
            
        Returns:
            logits: (batch_size, num_items)
        """
        # Base PQ scores
        base_logits = self.score_all_items(seq_emb)
        
        # Calculate bonuses
        pps_std, subid_std = self._calculate_bonuses(user_sequence_ids)
        
        # Combine: gamma * base + alpha * pps + beta * subid
        gamma = 1.0 - self.alpha - self.beta
        final_logits = gamma * base_logits + self.alpha * pps_std + self.beta * subid_std
        
        return final_logits

    def score_masked_tokens_with_bonus_user(self, masked_embeddings: torch.Tensor, user_sequence_ids: torch.Tensor):
        """
        Score all items with personalized bonus for masked tokens (for training).
        
        Args:
            masked_embeddings: (num_masked, embedding_size)
            user_sequence_ids: (num_masked, seq_len) or (num_masked,)
            
        Returns:
            logits: (num_masked, num_items)
        """
        # Base PQ scores
        base_logits = self.score_masked_tokens(masked_embeddings)
        
        # Calculate bonuses
        pps_std, subid_std = self._calculate_bonuses(user_sequence_ids)
        
        # Combine
        gamma = 1.0 - self.alpha - self.beta
        final_logits = gamma * base_logits + self.alpha * pps_std + self.beta * subid_std
        
        return final_logits
