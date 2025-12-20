from typing import Tuple, Optional

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix


class LazyInteractionDataset(Dataset):
    """A PyTorch Dataset that serves rows from a sparse matrix on-the-fly.

    This dataset is designed as a low-memory alternative to creating a dense tensor
    from a large user-item interaction matrix. It holds a reference to the sparse
    matrix and converts only one row (one user's interactions) to a dense tensor
    at a time when `__getitem__` is called.

    This avoids the massive memory allocation required by `sparse_matrix.todense()`.

    Args:
        sparse_matrix (csr_matrix): The user-item interaction matrix in CSR format.
        include_user_id (bool): If True, also returns the index of the user.
    """

    def __init__(self, sparse_matrix: csr_matrix, include_user_id: bool = False):
        self.sparse_matrix = sparse_matrix
        self.include_user_id = include_user_id

    def __len__(self) -> int:
        """Returns the total number of users in the matrix."""
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor] | Tuple[Tensor, Tensor]:
        """Retrieves one user's interaction vector as a dense tensor.

        Args:
            idx (int): The index of the user (row) to retrieve.

        Returns:
            Tuple[Tensor] | Tuple[Tensor, Tensor]: A Tuple containing 1D dense tensor representing
                the user's interactions with all items. If the user_id was requested it will
                also be returned.
        """
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
        else:
            # Normal behavior
            return (user_tensor,)


class LazyItemRatingDataset(Dataset):
    """A PyTorch Dataset for (user, item, rating) triplets that generates samples on-the-fly.

    This dataset is a low-memory alternative to pre-computing all positive and negative
    interactions. It calculates the total number of samples (positives + negatives)
    and maps any given index `idx` to either a positive interaction (rating=1.0) or a
    newly sampled negative interaction (rating=0.0).

    Args:
        user_ids (np.ndarray): The Numpy array of user ids aligned with the items.
        item_ids (np.ndarray): The Numpy array of item ids aligned with the users.
        sparse_matrix (csr_matrix): The user-item interaction matrix in CSR format.
        neg_samples (int): The number of negative samples to generate for each
            positive interaction.
        niid (int): The total number of unique items for negative sampling.
        seed (int): A random seed to ensure reproducibility of negative sampling.
        contexts (Optional[Tensor]): The tensor containing the context information
            of each interaction.

    Raises:
        ValueError: If arrays length mismatch.
    """

    def __init__(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        sparse_matrix: csr_matrix,
        neg_samples: int,
        niid: int,
        seed: int = 42,
        contexts: Optional[Tensor] = None,
    ):
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.sparse_matrix = sparse_matrix
        self.neg_samples = neg_samples
        self.niid = niid
        self.seed = seed
        self.contexts = contexts
        self.num_positives = len(self.user_ids)

        # Safety check
        if len(self.user_ids) != len(self.item_ids):
            raise ValueError("User and Item arrays must have the same length.")
        if self.contexts is not None and len(self.contexts) != self.num_positives:
            raise ValueError("Context tensor length mismatch.")

    def __len__(self) -> int:
        """Returns the total number of samples (positive + negative)."""
        # For each positive sample, we have 1 positive + neg_samples negatives
        return self.num_positives * (1 + self.neg_samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        """Generates and returns a single (user, item, rating) triplet.

        Args:
            idx (int): The index of the sample to generate.

        Returns:
            Tuple[Tensor, ...]: A tuple containing (user_id, item_id, rating).
        """
        # Map the indices
        pos_interaction_idx = idx // (1 + self.neg_samples)
        sample_offset = idx % (1 + self.neg_samples)

        # Retrieve the user
        user = self.user_ids[pos_interaction_idx]

        if sample_offset == 0:
            # Positive sample
            item = self.item_ids[pos_interaction_idx]
            rating = 1.0
        else:
            # Negative sample
            rating = 0.0

            # Define the set of seen items
            user_pos_set = set(self.sparse_matrix[user].indices)

            # Deterministic rng for reproducibility
            rng = np.random.default_rng(
                seed=(self.seed, pos_interaction_idx, sample_offset)
            )

            while True:
                candidate_item = rng.integers(0, self.niid)
                if candidate_item not in user_pos_set:
                    item = candidate_item
                    break

        # Convert to tensor
        user_tensor = torch.tensor(user, dtype=torch.long)
        item_tensor = torch.tensor(item, dtype=torch.long)
        rating_tensor = torch.tensor(rating, dtype=torch.float)

        # Handle context if required
        if self.contexts is not None:
            context_tensor = self.contexts[pos_interaction_idx]
            return user_tensor, item_tensor, rating_tensor, context_tensor

        return user_tensor, item_tensor, rating_tensor


class LazyTripletDataset(Dataset):
    """A PyTorch Dataset for (user, positive_item, negative_item) triplets.

    This dataset generates samples on-the-fly to serve as a low-memory alternative
    to pre-computing all possible triplets for BPR-style loss functions.

    For any given index `idx`, it identifies the `idx`-th positive interaction in the
    dataset and samples a corresponding negative item that the user has not
    interacted with.

    Args:
        sparse_matrix (csr_matrix): The user-item interaction matrix in CSR format.
        niid (int): The total number of unique items for negative sampling.
        seed (int): A random seed to ensure reproducibility of negative sampling.
    """

    def __init__(self, sparse_matrix: csr_matrix, niid: int, seed: int = 42):
        self.sparse_matrix = sparse_matrix
        self.niid = niid
        self.seed = seed

        # The COO format is the most efficient way to get a flat list of all
        # positive (user, item) pairs
        sparse_matrix_coo = self.sparse_matrix.tocoo()
        self.pos_users = sparse_matrix_coo.row
        self.pos_items = sparse_matrix_coo.col
        self.num_positives = self.sparse_matrix.nnz

    def __len__(self) -> int:
        """Returns the total number of positive interactions."""
        return self.num_positives

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Generates and returns a single (user, positive, negative) triplet.

        Args:
            idx (int): The index of the positive interaction to use for the triplet.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing (user_id, positive_item_id, negative_item_id).
        """
        # Get the positive (user, item) pair for the given index
        user = self.pos_users[idx]
        positive_item = self.pos_items[idx]

        # Sample a negative item.
        # Get the set of all items this user has interacted with for efficient collision checking
        user_pos_set = set(self.sparse_matrix[user].indices)

        # Use a unique, deterministic RNG for this specific triplet to ensure reproducibility
        rng = np.random.default_rng(seed=(self.seed, idx))

        # Keep sampling until a valid negative item (one not in the user's history) is found
        while True:
            negative_item = rng.integers(0, self.niid)
            if negative_item not in user_pos_set:
                break

        # Convert to tensors for the DataLoader
        user_tensor = torch.tensor(user, dtype=torch.long)
        positive_tensor = torch.tensor(positive_item, dtype=torch.long)
        negative_tensor = torch.tensor(negative_item, dtype=torch.long)

        return user_tensor, positive_tensor, negative_tensor
