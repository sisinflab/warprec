import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Union, Tuple, Optional


from ..centroid_assignment_strategies.centroid_strategy import CentroidAssignmentStragety
from ..centroid_assignment_strategies.svd_strategy import SVDAssignmentStrategy

def get_codes_strategy(codes_strategy: str, item_code_bytes: int, num_items: int) -> CentroidAssignmentStragety:
    if codes_strategy == "svd":
        return SVDAssignmentStrategy(item_code_bytes, num_items)
    if codes_strategy == "bpr":
        return BPRAssignmentStrategy(item_code_bytes, num_items)
    if codes_strategy == "random":
        return RandomAssignmentStrategy(item_code_bytes, num_items)
    if codes_strategy == "qr":
        return QuotientRemainder(item_code_bytes, num_items)
    raise ValueError(f"Unknown strategy: {codes_strategy}")

class ItemCodeLayer(nn.Module):
    def __init__(self, embedding_size, pq_m, num_items, sequence_length, codes_strategy, log_softmax=False, exp=False):
        super().__init__()
        self.sub_embedding_size = embedding_size // pq_m
        self.item_code_bytes = pq_m 
        self.sequence_length = sequence_length
        self.num_items = num_items
        self.log_softmax = log_softmax
        self.exp = exp

        if log_softmax and exp:
            raise ValueError("log_softmax and exp cannot be used together")

        self.vals_per_dim = 256 if codes_strategy != "qr" else math.ceil(math.sqrt(num_items))
        self.base_type = torch.uint8 if codes_strategy != "qr" else torch.int32

        # Buffers
        self.register_buffer(
            "item_codes", 
            torch.zeros((num_items + 2, self.item_code_bytes), dtype=self.base_type)
        )
        self.register_buffer(
            "codes_count",
            torch.ones((self.item_code_bytes, self.vals_per_dim), dtype=torch.float32)
        )

        # Trainable Centroids
        self.centroids = nn.Parameter(
            torch.empty((self.item_code_bytes, self.vals_per_dim, self.sub_embedding_size))
        )
        nn.init.uniform_(self.centroids, -0.05, 0.05)

        # --- DEBUG PRINT: EMBEDDING INITIALIZATION ---
        print(f"\n[DEBUG ItemCodeLayer] Centroids Initialized.")
        print(f"  > Shape: {self.centroids.shape} (Bytes, Vals, SubDim)")
        print(f"  > Mean: {self.centroids.mean().item():.6f}")
        print(f"  > Std:  {self.centroids.std().item():.6f}")
        print(f"  > Min/Max: {self.centroids.min().item():.4f} / {self.centroids.max().item():.4f}")
        # ---------------------------------------------

        if isinstance(codes_strategy, str):
            self.item_codes_strategy = get_codes_strategy(codes_strategy, self.item_code_bytes, num_items)
        elif isinstance(codes_strategy, CentroidAssignmentStragety):
            self.item_codes_strategy = codes_strategy
        else:
            raise TypeError("Invalid code assignment strategy")

    def pass_mapping(self, item_mappings):
        self.item_codes_strategy.pass_item_mappings(item_mappings)

    def assign_codes(self, train_users):
        codes = self.item_codes_strategy.assign(train_users)
        print(f"Assigned codes shape: {codes.shape}")
        
        # Copia i codici
        self.item_codes.copy_(torch.tensor(codes, dtype=self.base_type, device=self.item_codes.device))
        
        # Calcolo dei conteggi vettorizzato (molto più veloce del doppio for)
        counter = np.zeros((self.item_code_bytes, self.vals_per_dim), dtype=np.float32)
        npcodes = codes # Usiamo direttamente l'output della strategy
        
        for j in range(self.item_code_bytes):
            # Conta occorrenze di ogni valore (0-255) in questa colonna (byte)
            counts = np.bincount(npcodes[:, j], minlength=self.vals_per_dim)
            counter[j, :] = counts.astype(np.float32)
        
        self.codes_count.copy_(torch.tensor(counter, dtype=torch.float32, device=self.codes_count.device))
        print("Codes count updated successfully.")

    def map_subid_to_items(self) -> dict:
        codes_np = self.item_codes.cpu().numpy()
        mapping = {}
        for item_index in range(codes_np.shape[0] - 2):
            code = codes_np[item_index]
            for byte_idx, subid in enumerate(code):
                key = (byte_idx, int(subid))
                mapping.setdefault(key, []).append(item_index)
        return mapping

    def forward(self, input_ids: torch.Tensor, batch_size: int = None):
        input_codes = self.item_codes[input_ids.long()].long().detach()
        
        sub_embeddings = []
        for i in range(self.item_code_bytes):
            sub_emb = F.embedding(input_codes[..., i], self.centroids[i])
            sub_embeddings.append(sub_emb)
        
        seq_emb = torch.cat(sub_embeddings, dim=-1)
        return seq_emb

    def score_all_centroids(self, seq_emb, batch_size=None):
        if batch_size is None:
            batch_size = seq_emb.shape[0]
            
        sub_emb = seq_emb.view(batch_size, self.sequence_length, self.item_code_bytes, self.sub_embedding_size)
        scores = torch.einsum("bsie,ine->bsin", sub_emb, self.centroids)
        
        if self.log_softmax:
            return F.log_softmax(scores, dim=-1)
        if self.exp:
            return torch.exp(scores)
        return scores

    def get_target_codes(self, target_ids):
        target_ids = F.relu(target_ids).long()
        codes = self.item_codes[target_ids]
        return codes.long()

    def score_sequence_items(self, seq_emb, target_ids, batch_size=None):
        scores = self.score_all_centroids(seq_emb, batch_size)
        target_codes = self.get_target_codes(target_ids)

        if target_codes.dim() == 4 and scores.dim() == 4:
             scores = scores.unsqueeze(2)
             num_candidates = target_codes.shape[2]
             scores = scores.expand(-1, -1, num_candidates, -1, -1)
        
        target_codes_expanded = target_codes.unsqueeze(-1)
        sub_scores = scores.gather(-1, target_codes_expanded)
        return torch.sum(sub_scores.squeeze(-1), dim=-1)

    def score_emb_items(self, seq_emb, target_ids, batch_size=None):
        if batch_size is None:
            batch_size = seq_emb.shape[0]
            
        sub_emb = seq_emb.view(batch_size, 1, self.item_code_bytes, self.sub_embedding_size)
        scores = torch.einsum("bsie,ine->bsin", sub_emb, self.centroids)
        
        if self.log_softmax:
            scores = F.log_softmax(scores, dim=-1)
        elif self.exp:
            scores = torch.exp(scores)
            
        target_ids = F.relu(target_ids).long()
        target_codes = self.item_codes[target_ids].long()
        if target_codes.dim() != 3: 
             target_codes = target_codes.unsqueeze(1)

        sub_scores = scores.gather(-1, target_codes.unsqueeze(-1))
        return torch.sum(sub_scores.squeeze(-1), dim=-1).squeeze(1)

    def score_sequence_all_items(self, seq_emb, batch_size=None):
        if batch_size is None:
            batch_size = seq_emb.shape[0]

        # scores: [Batch, Seq, pq_m, 256]
        sub_emb = seq_emb.view(batch_size, self.sequence_length, self.item_code_bytes, self.sub_embedding_size)
        scores = torch.einsum("bsie,ine->bsin", sub_emb, self.centroids)
        
        if self.log_softmax:
            scores = F.log_softmax(scores, dim=-1)
        elif self.exp:
            scores = torch.exp(scores)
            
        # target_codes: [NumItems, pq_m]
        target_codes = self.item_codes[:-2].long()
        
        # Inizializziamo il tensore finale (3D): [Batch, Seq, NumItems]
        # Con B=128, S=150, I=29200, questo occupa circa 2.2 GB.
        final_scores = torch.zeros((batch_size, self.sequence_length, self.num_items), device=seq_emb.device)
        
        # CICLO SUI BYTE: Molto più efficiente in termini di memoria
        for i in range(self.item_code_bytes):
            # byte_scores: [Batch, Seq, 256]
            byte_scores = scores[:, :, i, :] 
            # item_indices: [NumItems]
            item_indices = target_codes[:, i]
            
            # index_select mappa i punteggi dei 256 centroidi sui 29200 item
            # partial_scores: [Batch, Seq, NumItems]
            final_scores += byte_scores.index_select(-1, item_indices)
            
        return final_scores
    
    def score_all_items(self, seq_emb):
        batch_size = seq_emb.shape[0]
        # sub_emb: [Batch, pq_m, sub_emb_size]
        sub_emb = seq_emb.view(batch_size, self.item_code_bytes, self.sub_embedding_size)
        scores = torch.einsum("bie,ine->bin", sub_emb, self.centroids)
        
        if self.log_softmax:
            scores = F.log_softmax(scores, dim=-1)
        elif self.exp:
            scores = torch.exp(scores)
            
        target_codes = self.item_codes[:-2].long()
        final_scores = torch.zeros((batch_size, self.num_items), device=seq_emb.device)
        
        for i in range(self.item_code_bytes):
            byte_scores = scores[:, i, :] # [Batch, 256]
            item_indices = target_codes[:, i] # [NumItems]
            
            final_scores += byte_scores.index_select(-1, item_indices)
        
        return final_scores

    def score_all_items_with_bonus_user(self, seq_emb: torch.Tensor, user_sequence_ids: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        base_logits = self.score_all_items(seq_emb)
        num_items = base_logits.shape[1]
        
        mask = (user_sequence_ids != -100)
        filtered_sequence_ids = user_sequence_ids[mask]
        unique_ids, counts = torch.unique(filtered_sequence_ids, return_counts=True)
        
        play_count_bonus = torch.zeros(num_items, device=seq_emb.device)
        play_count_bonus.index_add_(0, unique_ids.long(), counts.float())
        
        bonus_tensor = torch.log(play_count_bonus + eps).unsqueeze(0)
        
        mean = torch.mean(bonus_tensor, dim=1, keepdim=True)
        std = torch.std(bonus_tensor, dim=1, keepdim=True)
        bonus_standardized = (bonus_tensor - mean) / (std + eps)
        
        alpha = 0.5
        return alpha * bonus_standardized + (1 - alpha) * base_logits
    

    def score_masked_tokens(self, masked_embeddings):
        """
        masked_embeddings: [N_masked, Embedding_Size]
        Calcola i punteggi PQ solo per i token che contribuiscono alla loss.
        """
        num_masked = masked_embeddings.shape[0]
        
        # 1. Proiezione sui centroidi
        # sub_emb: [N_masked, pq_m, sub_embedding_size]
        sub_emb = masked_embeddings.view(num_masked, self.item_code_bytes, self.sub_embedding_size)
        
        # scores: [N_masked, pq_m, 256]
        scores = torch.einsum("mie,ine->min", sub_emb, self.centroids)
        
        if self.log_softmax:
            scores = F.log_softmax(scores, dim=-1)
            
        # 2. Mapping sugli item
        target_codes = self.item_codes[:-2].long() # [NumItems, pq_m]
        
        # final_scores: [N_masked, NumItems]
        final_scores = torch.zeros((num_masked, self.num_items), device=masked_embeddings.device)
        
        # Loop sui byte (pq_m). Molto più veloce perché N_masked è piccolo (~2800 invece di 19200)
        for i in range(self.item_code_bytes):
            byte_scores = scores[:, i, :] # [N_masked, 256]
            item_indices = target_codes[:, i] # [NumItems]
            
            # Somma in-place per risparmiare memoria
            final_scores.add_(byte_scores.index_select(-1, item_indices))
            
        return final_scores