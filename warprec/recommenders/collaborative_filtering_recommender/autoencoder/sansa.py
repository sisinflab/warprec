# pylint: disable = R0801, E1102
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from scipy.sparse import csr_matrix

try:
    from sksparse.cholmod import cholesky

    HAS_SKSPARSE = True
except ImportError:
    HAS_SKSPARSE = False

from warprec.recommenders.base_recommender import ItemSimRecommender
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="SANSA")
class SANSA(ItemSimRecommender):
    """Implementation of SANSA algorithm from
        "Scalable Approximate NonSymmetric Autoencoder forCollaborative Filtering" in RecSys 23.

    This model implements a sparse approximate inversion of the Gram matrix.
    It attempts to use `scikit-sparse` (CHOLMOD) for high-performance Cholesky
    factorization. If not available, it falls back to `scipy.sparse` inversion.

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3604915.3608827>`_.

    Args:
        params (dict): The dictionary with the model params.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Argument for PyTorch nn.Module.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Keyword argument for PyTorch nn.Module.

    Attributes:
        l2 (float): The L2 regularization value.
        target_density (float): The desired density of the weight matrix B.
    """

    l2: float
    target_density: float

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, seed=seed, **kwargs)

        X = interactions.get_sparse()

        # Compute Sparse Gram Matrix: G = X^T X
        G = (X.T @ X).tocsc()

        # Add L2 regularization to the diagonal
        diag_indices = np.arange(G.shape[0])
        G[diag_indices, diag_indices] += self.l2

        # Compute Approximate Inverse P = G^-1
        P = self._invert_matrix(G)

        # Extract diagonal of P
        diag_P = P.diagonal()

        # Handle division by zero (safety check)
        inv_diag = np.where(diag_P != 0, 1.0 / -diag_P, 0.0)

        # B = P * diag(inv_diag)
        # Scaling columns of P by inv_diag values.
        # Since P is symmetric (or close to), this implements the EASE formula.
        B = P.multiply(inv_diag[None, :])

        # Zero out the diagonal (constraint of the autoencoder)
        B.setdiag(0.0)
        B.eliminate_zeros()

        # Sparsification
        if self.target_density < 1.0:
            B = self._sparsify_matrix(B, self.target_density)

        # Store as CSR for fast multiplication during predict
        self.item_similarity = B.tocsr()

    def _invert_matrix(self, G: sp.csc_matrix) -> sp.csc_matrix:
        """Inverts the Gram matrix using the best available solver."""

        if HAS_SKSPARSE:
            try:
                # Factorize G = L D L^T
                factor = cholesky(G)
                # Solve G * P = I to get P
                # This is much faster and numerically stable than generic inversion
                P = factor.solve_A(sp.eye(G.shape[0], format="csc"))
                return P
            except Exception as e:
                print(
                    f"CHOLMOD failed: {e}"
                )  # If CHOLMOD fails, we will fall back to SciPy inversion

        # Fallback path: SciPy
        try:
            # Note: The inverse of a sparse matrix can have significant fill-in
            P = sp.linalg.inv(G)
            return P
        except RuntimeError:
            raise RuntimeError(
                "SANSA: Matrix inversion failed. The dataset might be too large "
                "or the matrix is singular."
            )

    def _sparsify_matrix(self, matrix: sp.spmatrix, density: float) -> sp.csr_matrix:
        """Retains only the top-k elements globally to achieve target density.

        Args:
            matrix (sp.spmatrix): The matrix to sparsify.
            density (float): The desired density ratio (0.0 to 1.0).

        Returns:
            sp.csr_matrix: The sparsified matrix.
        """
        if density >= 1.0:
            return matrix.tocsr()

        # Convert to COO to access data easily
        matrix_coo = matrix.tocoo()
        n_total = matrix.shape[0] * matrix.shape[1]
        k = int(n_total * density)

        # If the matrix is already sparser than the target, return it
        if matrix_coo.nnz <= k:
            return matrix.tocsr()

        # Find threshold value using numpy's partition (O(N) complexity)
        data_abs = np.abs(matrix_coo.data)

        # We want the indices of the top k elements.
        # argpartition puts the k-th largest element at index -k,
        # and all larger elements after it.
        partition_idx = np.argpartition(data_abs, -k)[-k]
        threshold = data_abs[partition_idx]

        # Filter elements below threshold
        mask = data_abs >= threshold

        # Create new sparse matrix
        new_data = matrix_coo.data[mask]
        new_row = matrix_coo.row[mask]
        new_col = matrix_coo.col[mask]

        return sp.csr_matrix((new_data, (new_row, new_col)), shape=matrix.shape)

    @torch.no_grad()
    def predict(
        self,
        train_batch: csr_matrix,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction override to handle Sparse Matrix multiplication.

        Args:
            train_batch (csr_matrix): The batch of user interaction vectors in sparse format.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix.
        """
        # Compute predictions: Sparse (Batch x Items) @ Sparse (Items x Items)
        # Result is Sparse (Batch x Items)
        predictions_sparse = train_batch @ self.item_similarity

        # Convert to dense Tensor
        predictions = torch.from_numpy(predictions_sparse.toarray()).float()

        # Return full or sampled predictions
        if item_indices is None:
            # Case 'full': prediction on all items
            return predictions  # [batch_size, n_items]

        # Case 'sampled': prediction on a sampled set of items
        return predictions.gather(
            1,
            item_indices.to(predictions.device).clamp(
                max=self.n_items - 1
            ),  # [batch_size, pad_seq]
        )
