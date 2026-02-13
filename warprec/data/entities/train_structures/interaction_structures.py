from typing import Tuple, Optional, List

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix


class InteractionDataset(Dataset):
    """A PyTorch Dataset that serves rows from a sparse matrix on-the-fly.

    This avoids the massive memory allocation required by `sparse_matrix.todense()`.

    Args:
        sparse_matrix (csr_matrix): The user-item interaction matrix in CSR format.
        include_user_id (bool): If True, also returns the index of the user.
    """

    def __init__(self, sparse_matrix: csr_matrix, include_user_id: bool = False):
        self.sparse_matrix = sparse_matrix
        self.include_user_id = include_user_id

    def __len__(self) -> int:
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        # CSR format is highly optimized for row slicing. This operation is very fast.
        user_row_sparse = self.sparse_matrix[idx]

        # Convert only this single row to a dense NumPy array.
        user_row_dense_np = user_row_sparse.todense()

        # Convert to a PyTorch tensor and remove the unnecessary leading dimension (shape [1, N] -> [N]).
        user_tensor = (
            torch.from_numpy(user_row_dense_np).to(dtype=torch.float32).squeeze(0)
        )

        if self.include_user_id:
            # Return also the user indices
            return torch.tensor(idx, dtype=torch.long), user_tensor

        # Normal behavior
        return (user_tensor,)


class PointWiseDataset(Dataset):
    """A PyTorch Dataset for (user, item, rating) triplets that generates samples on-the-fly.

    It calculates the total number of samples (positives + negatives)
    and maps any given index `idx` to either a positive interaction (rating=1.0) or a
    newly sampled negative interaction (rating=0.0).

    Args:
        user_ids (Tensor): The Torch tensor of user ids aligned with the items.
        item_ids (Tensor): The Torch tensor of item ids aligned with the users.
        sparse_matrix (csr_matrix): The user-item interaction matrix in CSR format.
        neg_samples (int): The number of negative samples to generate for each
            positive interaction.
        niid (int): The total number of unique items for negative sampling.
        side_information (Optional[Tensor]): The tensor containing the side information
            of each interaction.
        contexts (Optional[Tensor]): The tensor containing the context information
            of each interaction.
    """

    def __init__(
        self,
        user_ids: Tensor,
        item_ids: Tensor,
        sparse_matrix: csr_matrix,
        neg_samples: int,
        niid: int,
        side_information: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
    ):
        # Keep a copy of positive values
        self.user_ids = user_ids
        self.item_ids = item_ids

        # CSR matrix for faster lookup
        self.sparse_matrix = sparse_matrix

        self.neg_samples = neg_samples
        self.niid = niid
        self.side_information = side_information
        self.contexts = contexts

        self.num_positives = len(self.user_ids)
        self.total_samples = self.num_positives * (1 + self.neg_samples)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        # Linear mapping
        # idx 0 -> (pos_idx=0, offset=0) -> Positive
        # idx 1 -> (pos_idx=0, offset=1) -> Negative
        pos_interaction_idx = idx // (1 + self.neg_samples)
        sample_offset = idx % (1 + self.neg_samples)

        user_tensor = self.user_ids[pos_interaction_idx]

        if sample_offset == 0:
            item_tensor = self.item_ids[pos_interaction_idx]
            rating_val = 1.0
        else:
            rating_val = 0.0

            # Fast lookup with csr matrix
            # user_tensor is a 0-d tensor, we need its integer value for indexing
            user_idx = user_tensor.item()
            start = self.sparse_matrix.indptr[user_idx]
            end = self.sparse_matrix.indptr[user_idx + 1]
            seen_items = self.sparse_matrix.indices[start:end]

            # Negative sampling
            while True:
                # NumPy random is generally faster than torch.randint for single scalars
                candidate = np.random.randint(0, self.niid)

                # Fast check on sorted array (CSR indices are sorted by default)
                idx_ins = np.searchsorted(seen_items, candidate)
                if idx_ins < len(seen_items) and seen_items[idx_ins] == candidate:
                    continue

                item_tensor = torch.tensor(candidate, dtype=torch.long)
                break

        # Rating
        rating_tensor = torch.tensor(rating_val, dtype=torch.float)

        # Explicitly type the list to avoid mypy errors
        ret: List[Tensor] = [user_tensor, item_tensor, rating_tensor]

        # Side Info
        if self.side_information is not None:
            ret.append(self.side_information[item_tensor])

        # Context
        if self.contexts is not None:
            ret.append(self.contexts[pos_interaction_idx])

        return tuple(ret)


class ContrastiveDataset(Dataset):
    """A PyTorch Dataset for (user, positive_item, negative_item) triplets.

    Generates negative samples on-the-fly using the sparse interaction matrix.

    Args:
        user_ids (Tensor): Tensor of user indices for positive interactions.
        item_ids (Tensor): Tensor of item indices for positive interactions.
        sparse_matrix (csr_matrix): The user-item interaction matrix in CSR format.
        niid (int): Total number of items available.
    """

    def __init__(
        self,
        user_ids: Tensor,
        item_ids: Tensor,
        sparse_matrix: csr_matrix,
        niid: int,
    ):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.sparse_matrix = sparse_matrix
        self.niid = niid

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        # Retrieve positive interaction
        user_tensor = self.user_ids[idx]
        pos_item_tensor = self.item_ids[idx]

        # Negative Sampling
        # Retrieve seen items for this user using CSR slicing
        user_idx = user_tensor.item()
        start = self.sparse_matrix.indptr[user_idx]
        end = self.sparse_matrix.indptr[user_idx + 1]
        seen_items = self.sparse_matrix.indices[start:end]

        while True:
            # Sample a random item
            candidate = np.random.randint(0, self.niid)

            # Check if it is a true negative
            idx_ins = np.searchsorted(seen_items, candidate)
            if idx_ins < len(seen_items) and seen_items[idx_ins] == candidate:
                continue

            neg_item_tensor = torch.tensor(candidate, dtype=torch.long)
            break

        # Return triplet (user, pos, neg)
        return user_tensor, pos_item_tensor, neg_item_tensor
