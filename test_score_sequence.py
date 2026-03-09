#!/usr/bin/env python3
import torch
import torch.nn.functional as F

# Simula ItemCodeLayer parameters
batch_size = 2
seq_len = 3
embedding_size = 128
item_code_bytes = 32
sub_embedding_size = embedding_size // item_code_bytes  # 4
vals_per_dim = 256
num_targets = 2

print(f"batch_size={batch_size}, seq_len={seq_len}, embedding_size={embedding_size}")
print(f"item_code_bytes={item_code_bytes}, sub_embedding_size={sub_embedding_size}")

# Create dummy data
seq_emb = torch.randn(batch_size, seq_len, embedding_size)
centroids = torch.randn(item_code_bytes, vals_per_dim, sub_embedding_size)
item_codes = torch.randint(0, vals_per_dim, (1000, item_code_bytes))
target_ids = torch.randint(0, 1000, (batch_size, seq_len, num_targets))

print(f"\nseq_emb shape: {seq_emb.shape}")
print(f"centroids shape: {centroids.shape}")
print(f"target_ids shape: {target_ids.shape}")

# Tentativo 1: La nuova implementazione
print("\n=== TEST NEW IMPLEMENTATION ===")
try:
    # Reshape: (batch, seq_len, item_code_bytes, sub_embedding_size)
    sub_emb = seq_emb.view(batch_size, seq_len, item_code_bytes, sub_embedding_size)
    print(f"sub_emb shape: {sub_emb.shape}")
    
    # Compute scores for all byte positions and all centroid values
    scores = torch.einsum("bsie,ine->bsin", sub_emb, centroids)
    print(f"scores shape: {scores.shape}")  # Should be (batch, seq_len, item_code_bytes, vals_per_dim)
    
    # Get codes for target items
    target_codes = item_codes[F.relu(target_ids).long()].long()
    print(f"target_codes shape: {target_codes.shape}")  # Should be (batch, seq_len, num_targets, item_code_bytes)
    
    # Permute target_codes to (batch, seq_len, item_code_bytes, num_targets)
    target_codes_perm = target_codes.permute(0, 1, 3, 2)
    print(f"target_codes_perm shape: {target_codes_perm.shape}")
    
    # Gather for all byte positions at once
    sub_scores = torch.gather(
        scores.unsqueeze(3).expand(-1, -1, -1, num_targets, -1),
        4,
        target_codes_perm.unsqueeze(-1)
    ).squeeze(-1)
    print(f"sub_scores shape: {sub_scores.shape}")  # Should be (batch, seq, bytes, num_targets)
    
    # Sum over bytes
    logits = torch.sum(sub_scores, dim=2)
    print(f"logits shape: {logits.shape}")  # Should be (batch, seq_len, num_targets)
    print(f"logits sample:\n{logits[0]}")
    print("✓ NEW IMPLEMENTATION WORKS")
except Exception as e:
    print(f"✗ NEW IMPLEMENTATION FAILED: {e}")
    import traceback
    traceback.print_exc()
