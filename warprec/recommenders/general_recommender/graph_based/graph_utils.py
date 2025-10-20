from abc import ABC

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from scipy.sparse import coo_matrix
from torch.nn.init import xavier_normal_
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
        device: torch.device | str = "cpu",
    ) -> SparseTensor:
        """Get the normalized interaction matrix of users and items.

        Args:
            interaction_matrix (coo_matrix): The full interaction matrix in coo format.
            n_users (int): The number of users.
            n_items (int): The number of items.
            device (torch.device | str): Device to use for the adjacency matrix.

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
        # This is the format expected by LGConv
        adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            sparse_sizes=(n_users + n_items, n_users + n_items),
        ).to(device)

        # LGConv will handle the normalization
        # so there is no need to do it here
        return adj

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


class NGCFLayer(nn.Module):
    """Implementation of a single layer of NGCF propagation.
    - First term: GCN-like aggregation of neighbors.
    - Second term: Element-wise product capturing interaction between ego-embedding and aggregated neighbors.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        message_dropout (float): The dropout value.
    """

    def __init__(
        self, in_features: int, out_features: int, message_dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrices for the two terms
        self.W1 = nn.Parameter(torch.Tensor(in_features, out_features))
        self.W2 = nn.Parameter(torch.Tensor(in_features, out_features))

        # Biases for the two terms
        self.b1 = nn.Parameter(torch.Tensor(1, out_features))
        self.b2 = nn.Parameter(torch.Tensor(1, out_features))

        # LeakyReLU non-linearity and dropout layer
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=message_dropout)

        self.init_parameters()

    def init_parameters(self):
        xavier_normal_(self.W1.data)
        xavier_normal_(self.W2.data)
        nn.init.zeros_(self.b1.data)
        nn.init.zeros_(self.b2.data)

    def forward(self, ego_embeddings: Tensor, adj_matrix: SparseTensor) -> Tensor:
        """
        Performs a single NGCF propagation step.

        Args:
            ego_embeddings (Tensor): Current embeddings of all nodes (users + items).
            adj_matrix (SparseTensor): Normalized adjacency matrix (A_hat).

        Returns:
            Tensor: Propagated embeddings for the next layer.
        """
        laplacian_embeddings = adj_matrix.matmul(ego_embeddings)

        # First term: (A_hat + I) * E * W1 + b1
        first_term = (
            torch.matmul(ego_embeddings + laplacian_embeddings, self.W1) + self.b1
        )

        # Second term: (A_hat * E) element-wise product E * W2 + b2
        second_term = torch.mul(ego_embeddings, laplacian_embeddings)
        second_term = torch.matmul(second_term, self.W2) + self.b2

        # Combine terms, apply activation, dropout, and normalize
        output_embeddings = self.leaky_relu(first_term + second_term)
        output_embeddings = self.dropout(output_embeddings)
        output_embeddings = F.normalize(
            output_embeddings, p=2, dim=1
        )  # L2 Normalization

        return output_embeddings
