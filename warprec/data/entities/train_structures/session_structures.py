from typing import Tuple, Any

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix


class SequentialDataset(Dataset):
    """
    Standard Sequential dataset.
    Returns: (Sequence, Length, PosTarget, [NegTarget])
    """

    def __init__(
        self,
        flat_items: np.ndarray,
        flat_users: np.ndarray,
        user_offsets: np.ndarray,
        valid_target_indices: np.ndarray,
        sparse_matrix: csr_matrix,
        max_seq_len: int,
        neg_samples: int,
        niid: int,
        include_user_id: bool = False,
    ):
        self.flat_items = flat_items
        self.flat_users = flat_users
        self.user_offsets = user_offsets
        self.sample_indices = valid_target_indices
        self.sparse_matrix = sparse_matrix
        self.max_seq_len = max_seq_len
        self.neg_samples = neg_samples
        self.niid = niid
        self.include_user_id = include_user_id
        self.padding_token = niid

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        target_flat_idx = self.sample_indices[idx]
        user_idx = self.flat_users[target_flat_idx]
        pos_item = self.flat_items[target_flat_idx]

        # History ends just before target
        user_start_idx = self.user_offsets[user_idx]
        seq_end_idx = target_flat_idx
        seq_start_idx = max(user_start_idx, seq_end_idx - self.max_seq_len)

        seq_array = self.flat_items[seq_start_idx:seq_end_idx].copy()
        seq_len = len(seq_array)

        # Pad Sequence (Left-aligned data)
        seq_tensor = torch.full(
            (self.max_seq_len,), self.padding_token, dtype=torch.long
        )
        seq_tensor[:seq_len] = torch.from_numpy(seq_array)

        ret = [
            seq_tensor,
            torch.tensor(seq_len, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
        ]

        if self.include_user_id:
            ret.insert(0, torch.tensor(user_idx, dtype=torch.long))

        # Negative Sampling
        if self.neg_samples > 0:
            neg_items: list[Any] = []
            u_start = self.sparse_matrix.indptr[user_idx]
            u_end = self.sparse_matrix.indptr[user_idx + 1]
            seen_items = self.sparse_matrix.indices[u_start:u_end]

            while len(neg_items) < self.neg_samples:
                cand = np.random.randint(0, self.niid)
                # Fast check on sorted CSR indices
                idx_ins = np.searchsorted(seen_items, cand)
                if idx_ins < len(seen_items) and seen_items[idx_ins] == cand:
                    continue
                if cand == pos_item:
                    continue
                neg_items.append(cand)

            neg_tensor = torch.tensor(neg_items, dtype=torch.long)
            if self.neg_samples == 1:
                neg_tensor = neg_tensor.squeeze(0)
            ret.append(neg_tensor)

        return tuple(ret)


class SlidingWindowDataset(Dataset):
    """
    Dataset for Sequence-to-Sequence training.
    Returns: (InputSequence, NegativeSamplesMatrix)
    """

    def __init__(
        self,
        flat_items: np.ndarray,
        window_starts: np.ndarray,
        window_users: np.ndarray,
        sparse_matrix: csr_matrix,
        max_seq_len: int,
        neg_samples: int,
        niid: int,
    ):
        self.flat_items = flat_items
        self.window_starts = window_starts
        self.window_users = window_users
        self.sparse_matrix = sparse_matrix
        self.max_seq_len = max_seq_len
        self.neg_samples = neg_samples
        self.niid = niid
        self.padding_token = niid

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        start_idx = self.window_starts[idx]
        user_idx = self.window_users[idx]

        # Clip end to user boundary
        user_end_limit = self.sparse_matrix.indptr[user_idx + 1]
        end_idx = min(start_idx + self.max_seq_len, user_end_limit)

        seq_array = self.flat_items[start_idx:end_idx]
        real_len = len(seq_array)

        # Input Sequence
        pos_seq = torch.full((self.max_seq_len,), self.padding_token, dtype=torch.long)
        pos_seq[:real_len] = torch.from_numpy(seq_array)

        if self.neg_samples > 0:
            neg_seq = torch.full(
                (self.max_seq_len, self.neg_samples),
                self.padding_token,
                dtype=torch.long,
            )

            u_start = self.sparse_matrix.indptr[user_idx]
            u_end = self.sparse_matrix.indptr[user_idx + 1]
            seen_items = self.sparse_matrix.indices[u_start:u_end]

            # Generate negatives for each valid time step
            for t in range(real_len):
                found = 0
                while found < self.neg_samples:
                    needed = self.neg_samples - found
                    candidates = np.random.randint(0, self.niid, size=needed)

                    # Vectorized check
                    idxs = np.searchsorted(seen_items, candidates)
                    idxs = np.clip(idxs, 0, len(seen_items) - 1)
                    is_seen = seen_items[idxs] == candidates

                    valid_cands = candidates[~is_seen]

                    num_valid = len(valid_cands)
                    if num_valid > 0:
                        neg_seq[t, found : found + num_valid] = torch.from_numpy(
                            valid_cands
                        )
                        found += num_valid

            return pos_seq, neg_seq

        return (pos_seq,)


class ClozeDataset(Dataset):
    """
    Dataset for Cloze Mask training.
    Returns 4 tensors: (MaskedSeq, PosItems, NegItems, MaskedIndices)
    """

    def __init__(
        self,
        flat_items: np.ndarray,
        window_starts: np.ndarray,
        window_ends: np.ndarray,
        window_users: np.ndarray,
        sparse_matrix: csr_matrix,
        max_seq_len: int,
        mask_prob: float,
        mask_token_id: int,
        neg_samples: int,
        niid: int,
        seed: int = 42,
    ):
        self.flat_items = flat_items
        self.window_starts = window_starts
        self.window_ends = window_ends
        self.window_users = window_users
        self.sparse_matrix = sparse_matrix
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.neg_samples = neg_samples
        self.niid = niid
        self.padding_token = niid
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        start = self.window_starts[idx]
        end = self.window_ends[idx]
        user_idx = self.window_users[idx]

        seq_array = self.flat_items[start:end].copy()
        real_seq_len = len(seq_array)

        # Masking Logic
        num_to_mask = max(1, int(real_seq_len * self.mask_prob))
        masked_indices = np.random.choice(real_seq_len, size=num_to_mask, replace=False)

        pos_targets = seq_array[masked_indices]
        seq_array[masked_indices] = self.mask_token_id

        # Negative Sampling (Only for masked items)
        neg_targets = np.full(
            (num_to_mask, self.neg_samples), self.padding_token, dtype=np.int64
        )

        if self.neg_samples > 0:
            u_start = self.sparse_matrix.indptr[user_idx]
            u_end = self.sparse_matrix.indptr[user_idx + 1]
            seen_items = self.sparse_matrix.indices[u_start:u_end]

            for i in range(num_to_mask):
                true_item = pos_targets[i]
                found_count = 0
                while found_count < self.neg_samples:
                    cand = np.random.randint(0, self.niid)
                    if cand == true_item:
                        continue

                    idx_ins = np.searchsorted(seen_items, cand)
                    if idx_ins < len(seen_items) and seen_items[idx_ins] == cand:
                        continue

                    neg_targets[i, found_count] = cand
                    found_count += 1

        # Tensor Construction (Compacted/Dense format for targets)

        # Masked Sequence [Item, Mask, Pad...]
        masked_seq_tensor = torch.full(
            (self.max_seq_len,), self.padding_token, dtype=torch.long
        )
        masked_seq_tensor[:real_seq_len] = torch.from_numpy(seq_array)

        # Positive Items [Target1, Target2, Pad...]
        pos_items_tensor = torch.full(
            (self.max_seq_len,), self.padding_token, dtype=torch.long
        )
        pos_items_tensor[:num_to_mask] = torch.from_numpy(pos_targets)

        # Negative Items [Negs1, Negs2, Pad...]
        neg_items_tensor = torch.full(
            (self.max_seq_len, self.neg_samples), self.padding_token, dtype=torch.long
        )
        neg_items_tensor[:num_to_mask, :] = torch.from_numpy(neg_targets)

        # Masked Indices [Idx1, Idx2, 0, 0...]
        masked_indices_tensor = torch.zeros(self.max_seq_len, dtype=torch.long)
        masked_indices_tensor[:num_to_mask] = torch.from_numpy(masked_indices)

        return (
            masked_seq_tensor,
            pos_items_tensor,
            neg_items_tensor,
            masked_indices_tensor,
        )
