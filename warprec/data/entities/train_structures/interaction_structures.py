from typing import Tuple

import torch
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
    """

    def __init__(self, sparse_matrix: csr_matrix):
        self.sparse_matrix = sparse_matrix

    def __len__(self) -> int:
        """Returns the total number of users in the matrix."""
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor]:
        """Retrieves one user's interaction vector as a dense tensor.

        Args:
            idx (int): The index of the user (row) to retrieve.

        Returns:
            Tuple[Tensor]: A Tuple containing 1D dense tensor representing
                the user's interactions with all items.
        """
        # CSR format is highly optimized for row slicing. This operation is very fast.
        user_row_sparse = self.sparse_matrix[idx]

        # Convert only this single row to a dense NumPy array.
        user_row_dense_np = user_row_sparse.todense()

        # Convert to a PyTorch tensor and remove the unnecessary leading dimension (shape [1, N] -> [N]).
        user_tensor = (
            torch.from_numpy(user_row_dense_np).to(dtype=torch.float32).squeeze(0)
        )

        return (user_tensor,)  # To mimic the behavior of a TensorDataset
