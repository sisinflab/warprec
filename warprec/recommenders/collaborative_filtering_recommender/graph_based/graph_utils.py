from typing import Any, Optional, Tuple

import torch
import numpy as np
from torch import nn, Tensor
from scipy.sparse import coo_matrix
from torch_geometric import EdgeIndex


class SparseAdjacency:
    """A sparse adjacency matrix backed by a PyG :class:`EdgeIndex`.

    This exposes the subset of the ``torch_sparse.SparseTensor`` interface that
    the graph recommenders use, so that the dependency can be dropped without
    rewriting every model. ``torch_sparse`` is only distributed from PyG's own
    wheel index, which cannot be expressed as a normal dependency, while
    ``EdgeIndex`` ships with ``torch_geometric`` on PyPI and is measurably
    faster on both CPU and GPU.

    Args:
        row (Tensor): Row indices of the non-zero entries.
        col (Tensor): Column indices of the non-zero entries.
        value (Optional[Tensor]): Values of the non-zero entries. When omitted
            the matrix is unweighted and every entry is 1, matching how the
            previous backend treated a matrix built without values.
        size (Tuple[int, int]): Shape of the matrix.
        is_sorted (bool): Whether the entries are already sorted by row and then
            by column. The underlying matmul requires that ordering, so leave
            this False unless the caller has already sorted them.
    """

    def __init__(
        self,
        row: Tensor,
        col: Tensor,
        value: Optional[Tensor] = None,
        size: Tuple[int, int] = None,
        is_sorted: bool = False,
    ):
        if value is None:
            value = torch.ones(row.numel(), dtype=torch.get_default_dtype())

        if not is_sorted:
            order = torch.argsort(row * size[1] + col)
            row, col, value = row[order], col[order], value[order]

        self._row = row
        self._col = col
        self._value = value
        self._size = size
        self._edge_index = EdgeIndex(
            torch.stack([row, col]), sparse_size=size, sort_order="row"
        )

    def matmul(self, other: Tensor, reduce: str = "sum") -> Tensor:
        """Multiplies this matrix by a dense matrix.

        Args:
            other (Tensor): The dense right-hand side.
            reduce (str): The reduction to apply.

        Returns:
            Tensor: The product.
        """
        return self._edge_index.matmul(other, input_value=self._value, reduce=reduce)

    def t(self) -> "SparseAdjacency":
        """Returns the transpose.

        Returns:
            SparseAdjacency: A new adjacency with rows and columns swapped,
                re-sorted so that it can be multiplied.
        """
        return SparseAdjacency(
            self._col, self._row, self._value, (self._size[1], self._size[0])
        )

    def coo(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns the matrix in coordinate form.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Rows, columns and values.
        """
        return self._row, self._col, self._value

    def sum(self, dim: int = 1) -> Tensor:
        """Sums the values along one dimension, giving the weighted degree.

        Args:
            dim (int): 1 to sum along rows, 0 to sum along columns.

        Returns:
            Tensor: The per-node sum.
        """
        length = self._size[0] if dim == 1 else self._size[1]
        index = self._row if dim == 1 else self._col
        out = torch.zeros(length, device=self._value.device, dtype=self._value.dtype)
        return out.scatter_add_(0, index, self._value)

    def set_value(self, value: Tensor, layout: str = "coo") -> "SparseAdjacency":
        """Returns a copy carrying different values for the same structure.

        Args:
            value (Tensor): The new values, in the order returned by 'coo'.
            layout (str): Accepted for compatibility with the previous backend
                and ignored, since values are always kept in coordinate order.

        Returns:
            SparseAdjacency: The updated adjacency.
        """
        return SparseAdjacency(self._row, self._col, value, self._size, is_sorted=True)

    def to(self, device: Any) -> "SparseAdjacency":
        """Moves the adjacency to a device.

        Args:
            device (Any): The target device.

        Returns:
            SparseAdjacency: The adjacency on the requested device.
        """
        return SparseAdjacency(
            self._row.to(device),
            self._col.to(device),
            self._value.to(device),
            self._size,
            is_sorted=True,
        )

    def device(self) -> torch.device:
        """Returns the device holding the adjacency.

        Returns:
            torch.device: The device.
        """
        return self._value.device

    def size(self, dim: int) -> int:
        """Returns the extent of one dimension.

        Args:
            dim (int): The dimension to query.

        Returns:
            int: The size of that dimension.
        """
        return self._size[dim]

    def sparse_sizes(self) -> Tuple[int, int]:
        """Returns the shape of the matrix.

        Returns:
            Tuple[int, int]: The number of rows and columns.
        """
        return self._size

    def nnz(self) -> int:
        """Returns the number of stored entries.

        Returns:
            int: The number of non-zero entries.
        """
        return int(self._value.numel())


class GraphRecommenderUtils(nn.Module):
    """Common definition for graph recommenders.

    Collection of common method used by all graph recommenders.
    """

    # Cache storage
    _cached_user_emb: Optional[Tensor]
    _cached_item_emb: Optional[Tensor]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize local cache
        self._cached_user_emb = None
        self._cached_item_emb = None

    def train(self, mode=True):
        """Override train mode to empty the cache when switching to training."""
        super().train(mode)

        if mode:
            # We are in training mode, embeddings will change. Empty the cache
            self._cached_user_emb = None
            self._cached_item_emb = None

    def eval(self):
        """Override eval mode to empty the cache when switching to evaluation."""
        super().eval()

        # We are in evaluation mode, embeddings will be cached. Empty the cache
        self._cached_user_emb = None
        self._cached_item_emb = None

    def propagate_embeddings(self) -> Tuple[Tensor, Tensor]:
        """Retrieve the propagate user and item embeddings.

        Subsequent calls will return the cached values, speeding up the
        evaluation process.

        Returns:
            Tuple[Tensor, Tensor]: (User Embeddings, Item Embeddings)
        """
        # Safety check
        if self.training:
            return self.forward()[:2]

        # Check if values are cached
        if self._cached_user_emb is None or self._cached_item_emb is None:
            with torch.no_grad():
                # Unpack the forward
                ret = self.forward()
                self._cached_user_emb = ret[0]
                self._cached_item_emb = ret[1]

        return self._cached_user_emb, self._cached_item_emb

    def get_adj_mat(
        self,
        interaction_matrix: coo_matrix,
        n_users: int,
        n_items: int,
        normalize: bool = False,
    ) -> SparseAdjacency:
        """Get the normalized interaction matrix of users and items.

        Args:
            interaction_matrix (coo_matrix): The full interaction matrix in coo format.
            n_users (int): The number of users.
            n_items (int): The number of items.
            normalize (bool): Wether or not to normalize the sparse adjacency matrix.

        Returns:
            SparseAdjacency: The sparse adjacency matrix.
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

        # Create the adjacency using the edge indexes. The previous backend
        # treated a valueless matrix as unweighted, so the values default to 1.
        adj = SparseAdjacency(
            row=edge_index[0],
            col=edge_index[1],
            size=(n_users + n_items, n_users + n_items),
        )

        # Normalize the adjacency if requested
        if normalize:
            adj = self._symmetric_normalization(adj)

        return adj

    def _symmetric_normalization(self, adj: SparseAdjacency) -> SparseAdjacency:
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
        if not 0 <= p <= 1:
            raise ValueError(
                f"Dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p

    def forward(self, X: SparseAdjacency) -> SparseAdjacency:
        """Apply dropout to the values of a sparse adjacency.

        Args:
            X (SparseAdjacency): The input adjacency.

        Returns:
            SparseAdjacency: The adjacency with dropped-out values, rescaled so
                that the expected value is unchanged.
        """
        if self.p == 0 or not self.training:
            return X

        _, _, values = X.coo()
        keep = (torch.rand(values.numel(), device=values.device) > self.p).to(
            values.dtype
        )
        return X.set_value(values * keep / (1 - self.p))
