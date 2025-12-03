from abc import ABC

import torch
import numpy as np
from torch import nn, Tensor
from scipy.sparse import coo_matrix
from torch_sparse import SparseTensor


class GraphRecommenderUtils(ABC):
    """Common definition for graph recommenders.

    Collection of common method used by all graph recommenders.
    """

    def get_adj_mat(
        self,
        interaction_matrix: coo_matrix,
        n_users: int,
        n_items: int,
        normalize: bool = False,
    ) -> SparseTensor:
        """Get the normalized interaction matrix of users and items.

        Args:
            interaction_matrix (coo_matrix): The full interaction matrix in coo format.
            n_users (int): The number of users.
            n_items (int): The number of items.
            normalize (bool): Wether or not to normalize the sparse adjacency matrix.

        Returns:
            SparseTensor: The sparse adjacency matrix.
        """
        # Extract user and items nodes
        user_nodes = interaction_matrix.row
        item_nodes = interaction_matrix.col + n_users

        # Unify arcs in both directions
        row = np.concatenate([user_nodes, item_nodes])
        col = np.concatenate([item_nodes, user_nodes])

        # Create the edge tensor
        edge_index_np = np.vstack([row, col])  # Efficient solution

        # Creating a tensor directly from a numpy array instead of lists
        edge_index = torch.tensor(edge_index_np, dtype=torch.int64)

        # Create the SparseTensor using the edge indexes.
        adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            sparse_sizes=(n_users + n_items, n_users + n_items),
        )

        # Normalize the SparseTensor if requested
        if normalize:
            adj = self._symmetric_normalization(adj)

        return adj

    def _symmetric_normalization(self, adj: SparseTensor) -> SparseTensor:
        """Applies symmetric normalization: D^-0.5 * A * D^-0.5."""
        # Calculate degree (sum of rows)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)

        # Apply normalization efficiently on the values
        row, col, _ = adj.coo()
        norm_vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return adj.set_value(norm_vals, layout="coo")

    def get_ego_embeddings(
        self, user_embedding: nn.Embedding, item_embedding: nn.Embedding
    ) -> Tensor:
        """Get the initial embedding of users and items and combine to an embedding matrix.

        Args:
            user_embedding (nn.Embedding): The user embeddings.
            item_embedding (nn.Embedding): The item embeddings.

        Returns:
            Tensor: Combined user and item embeddings.
        """
        user_embeddings = user_embedding.weight
        item_embeddings = item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings


class SparseDropout(nn.Module):
    """Dropout layer for sparse tensors.

    Args:
        p (float): Dropout rate. Values accepted in range [0, 1].

    Raises:
        ValueError: If p is not in range.
    """

    def __init__(self, p: float):
        super().__init__()
        if not (0 <= p <= 1):
            raise ValueError(
                f"Dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p

    def forward(self, X: SparseTensor) -> SparseTensor:
        """Apply dropout to SparseTensor.

        Args:
            X (SparseTensor): The input tensor.

        Returns:
            SparseTensor: The tensor after the dropout.
        """
        if self.p == 0 or not self.training:
            return X

        # Get indices and values of the sparse tensor
        indices = X.indices()
        values = X.values()

        # Calculate number of non-zero elements
        n_nonzero_elems = values.numel()

        # Create a dropout mask
        random_tensor = torch.rand(n_nonzero_elems, device=X.device)
        dropout_mask = (random_tensor > self.p).to(X.dtype)

        # Apply mask and scale
        out_values = values * dropout_mask / (1 - self.p)

        # Return the tensor as a SparseTensor in coo format
        return torch.sparse_coo_tensor(indices, out_values, X.size(), device=X.device)
