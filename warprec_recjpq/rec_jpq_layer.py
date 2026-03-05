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

        self.register_buffer(
            "item_codes",
            torch.zeros((num_items + 2, self.item_code_bytes), dtype=self.base_type),
        )
        self.register_buffer(
            "codes_count",
            torch.ones((self.item_code_bytes, self.vals_per_dim), dtype=torch.float32),
        )

        self.centroids = nn.Parameter(
            torch.empty(
                (self.item_code_bytes, self.vals_per_dim, self.sub_embedding_size)
            )
        )
        nn.init.uniform_(self.centroids, -0.05, 0.05)

         
        print(f"\n[DEBUG ItemCodeLayer] Centroids Initialized.")
        print(f"  > Shape: {self.centroids.shape} (Bytes, Vals, SubDim)")
        print(f"  > Mean: {self.centroids.mean().item():.6f}")
        print(f"  > Std:  {self.centroids.std().item():.6f}")
        print(f"  > Min/Max: {self.centroids.min().item():.4f} / {self.centroids.max().item():.4f}")
      

        if isinstance(codes_strategy, str):
            self.item_codes_strategy = get_codes_strategy(
                codes_strategy, self.item_code_bytes, num_items
            )
        elif isinstance(codes_strategy, CentroidAssignmentStragety):
            self.item_codes_strategy = codes_strategy
        else:
            raise TypeError("Invalid code assignment strategy")

    def pass_mapping(self, item_mappings):
        self.item_codes_strategy.pass_item_mappings(item_mappings)

    def assign_codes(self, train_users):
        codes = self.item_codes_strategy.assign(train_users)
        self.item_codes.copy_(
            torch.tensor(codes, dtype=self.base_type, device=self.item_codes.device)
        )

        counter = np.zeros((self.item_code_bytes, self.vals_per_dim), dtype=np.float32)
        for dim_idx in range(self.item_code_bytes):
            counts = np.bincount(codes[:, dim_idx], minlength=self.vals_per_dim)
            counter[dim_idx, :] = counts.astype(np.float32)

        self.codes_count.copy_(
            torch.tensor(
                counter, dtype=torch.float32, device=self.codes_count.device
            )
        )

    def forward(self, input_ids: torch.Tensor):
        input_codes = self.item_codes[input_ids.long()].long().detach()
        sub_embeddings = []
        for dim_idx in range(self.item_code_bytes):
            sub_embeddings.append(F.embedding(input_codes[..., dim_idx], self.centroids[dim_idx]))
        return torch.cat(sub_embeddings, dim=-1)

    def score_emb_items(self, seq_emb: torch.Tensor, target_ids: torch.Tensor):
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

        num_candidates = target_codes.shape[1]
        if scores.shape[1] != num_candidates:
            scores = scores.expand(-1, num_candidates, -1, -1)

        sub_scores = scores.gather(-1, target_codes.unsqueeze(-1))
        return torch.sum(sub_scores.squeeze(-1), dim=-1).squeeze(1)

    def score_all_items(self, seq_emb: torch.Tensor):
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
