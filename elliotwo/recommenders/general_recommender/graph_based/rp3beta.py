# pylint: disable = R0801, E1102
from typing import Union, Optional, Callable, Any

import torch
import numpy as np
from torch import Tensor, nn
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from sklearn.preprocessing import normalize
from elliotwo.data.dataset import Interactions
from elliotwo.recommenders.base_recommender import Recommender
from elliotwo.utils.registry import model_registry


@model_registry.register(name="RP3Beta")
class RP3Beta(Recommender):
    """Implementation of RP3Beta algorithm from
        Updatable, accurate, diverse, and scalable recommendations for interactive applications 2016.

    For further details, check the `paper <https://www.zora.uzh.ch/id/eprint/131338/1/TiiS_2016.pdf>`_.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items value was not passed through the info dict.

    Attributes:
        k (int): Number of nearest neighbors.
        alpha (float): The intensity of the normalization.
        beta (float): The normalization value for the users connections.
        normalize (bool): Wether or not to normalize the interactions.
    """

    k: int
    alpha: float
    beta: float
    normalize: bool

    def __init__(
        self,
        params: dict,
        *args: Any,
        device: str = "cpu",
        seed: int = 42,
        info: dict = None,
        **kwargs: Any,
    ):
        super().__init__(params, device=device, seed=seed, *args, **kwargs)
        self._name = "RP3Beta"
        items = info.get("items", None)
        if not items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        # Model initialization
        self.item_similarity = nn.Parameter(torch.rand(items, items)).to(self._device)

    def fit(
        self,
        interactions: Interactions,
        report_fn: Optional[Callable] = None,
        *args: Any,
        **kwargs: Any,
    ):
        """Main train method.

        The training will be conducted on the sparse representation of the interactions.
        During the train a similarity matrix {item x item} will be learned.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        X = interactions.get_sparse()

        # Step 1: Normalize user-item matrix
        Pui = normalize(X, norm="l1", axis=1)

        # Step 2: Create boolean item-user matrix
        X_bool = X.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Step 3: Calculate item popularity degrees
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()
        degree = np.zeros(X.shape[1])
        non_zero_mask = X_bool_sum != 0.0
        degree[non_zero_mask] = np.power(X_bool_sum[non_zero_mask], -self.beta)

        # Step 4: Normalize item-user matrix
        Piu = normalize(X_bool, norm="l1", axis=1)

        # Apply alpha exponent
        if self.alpha != 1.0:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        # Step 5: Compute similarity in blocks
        similarity_matrix = self._compute_blockwise_similarity(Piu, Pui, degree)

        # Step 6: Apply top-k filtering
        filtered_matrix = self._apply_sparse_topk(similarity_matrix, self.k)

        # Step 7: Normalize if required
        if self.normalize:
            filtered_matrix = normalize(filtered_matrix, norm="l1", axis=1)

        # Update item_similarity with a new nn.Parameter, doing the proper casting
        self.item_similarity = nn.Parameter(
            torch.from_numpy(filtered_matrix.toarray()).to(self._device)
        )

        if report_fn is not None:
            report_fn(self)

    def _compute_blockwise_similarity(
        self,
        Piu: csr_matrix,
        Pui: csr_matrix,
        degree: csr_matrix,
        initial_block_dim: int = 200,
        initial_data_block: int = 10000000,
    ) -> csr_matrix:
        """
        Computes similarity matrix in blocks to handle large matrices efficiently.

        Args:
            Piu (csr_matrix): User-item interaction matrix.
            Pui (csr_matrix): Item-user interaction matrix.
            degree (csr_matrix): Diagonal matrix containing degree information for normalization.
            initial_block_dim (int): Initial dimension for row blocking.
            initial_data_block (int): Initial storage allocation for non-zero elements.

        Returns:
            csr_matrix: Computed similarity matrix between items (items x items)
        """
        block_dim = initial_block_dim
        data_block = initial_data_block

        # Initialize storage arrays with initial allocation
        rows = np.zeros(data_block, dtype=np.int32)
        cols = np.zeros(data_block, dtype=np.int32)
        values = np.zeros(data_block, dtype=np.float32)
        num_cells = 0

        # Process matrix in blocks along rows
        for current_block_start_row in range(0, Pui.shape[1], block_dim):
            # Adjust block dimension for last block
            block_dim = min(block_dim, Pui.shape[1] - current_block_start_row)

            # Compute similarity block matrix product
            similarity_block = (
                Piu[current_block_start_row : current_block_start_row + block_dim] @ Pui
            )
            similarity_block = similarity_block.multiply(degree).tocoo()

            # Remove self-similarity entries (diagonal elements)
            mask = (
                similarity_block.row + current_block_start_row
            ) != similarity_block.col
            similarity_block.row = similarity_block.row[mask] + current_block_start_row
            similarity_block.col = similarity_block.col[mask]
            similarity_block.data = similarity_block.data[mask]

            # Check if we need to expand storage
            new_entries = len(similarity_block.data)
            while num_cells + new_entries > len(rows):
                # Expand storage arrays exponentially
                rows = np.concatenate((rows, np.zeros(data_block, dtype=np.int32)))  # type: ignore
                cols = np.concatenate((cols, np.zeros(data_block, dtype=np.int32)))  # type: ignore
                values = np.concatenate(
                    (values, np.zeros(data_block, dtype=np.float32))
                )  # type: ignore

            # Store computed values
            rows[num_cells : num_cells + new_entries] = similarity_block.row
            cols[num_cells : num_cells + new_entries] = similarity_block.col
            values[num_cells : num_cells + new_entries] = similarity_block.data
            num_cells += new_entries

        # Create final sparse matrix from accumulated values
        return csr_matrix(
            (values[:num_cells], (rows[:num_cells], cols[:num_cells])),
            shape=(Pui.shape[1], Pui.shape[1]),
        )

    def _apply_sparse_topk(
        self, matrix: Union[csr_matrix, coo_matrix], k: int
    ) -> csr_matrix:
        """
        Applies top-k filtering to each row of a sparse matrix efficiently.

        Args:
            matrix (Union[csr_matrix, coo_matrix]): Input sparse matrix.
            k (int): Number of maximum values to preserve per row.

        Returns:
            csr_matrix: Sparse matrix with only the top-k elements per row preserved
        """
        filtered_matrix: lil_matrix = lil_matrix(matrix.shape, dtype=np.float32)

        # Process each row individually
        for i in range(matrix.shape[0]):
            row: coo_matrix = matrix[i].tocoo()
            if row.nnz == 0:  # Skip empty rows
                continue

            # Determine actual number of elements to keep
            top_k: int = min(k, row.nnz)

            # Find indices of top-k largest values using argpartition
            idx: np.ndarray = np.argpartition(row.data, -top_k)[-top_k:]

            # Store top-k values in their original column positions
            filtered_matrix[i, row.col[idx]] = row.data[idx]

        # Convert to CSR format for efficient subsequent operations
        return filtered_matrix.tocsr()

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction in the form of X@B where B is a {item x item} similarity matrix.

        Args:
            interaction_matrix (csr_matrix): The interactions matrix
                that will be used to predict.
            *args (Any): List of arguments.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        r = interaction_matrix @ self.item_similarity.detach().numpy()

        # Masking interaction already seen in train
        r[interaction_matrix.nonzero()] = -torch.inf
        return torch.from_numpy(r).to(self._device)

    def forward(self, *args, **kwargs):
        """Forward method is empty because we don't need
        back propagation.
        """
        pass
