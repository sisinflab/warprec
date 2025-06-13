from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_
from torch_sparse import SparseTensor

from warprec.utils.enums import Activations, Initializations


class MLP(nn.Module):
    """Simple implementation of MultiLayer Perceptron.

    Args:
        layers (List[int]): The hidden layers size list.
        dropout (float): The dropout probability.
        activation (Activations): The activation function to apply.
        batch_normalization (bool): Wether or not to apply batch normalization.
        initialization_method (Initializations): The method of initialization to use.
        last_activation (bool): Wether or not to keep last non-linearity function.
    """

    def __init__(
        self,
        layers: List[int],
        dropout: float = 0.0,
        activation: Activations = Activations.RELU,
        batch_normalization: bool = False,
        initialization_method: Initializations = None,
        last_activation: bool = True,
    ):
        super().__init__()
        self.init = initialization_method
        mlp_modules: List[Module] = []
        for input_size, output_size in zip(layers[:-1], layers[1:]):
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if batch_normalization:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            if activation:
                mlp_modules.append(self._get_activation(activation))
        if activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if initialization_method:
            self.apply(self._init_weights)

    def forward(self, input_feature: Tensor):
        """Simple forwarding, input tensor will pass
        through all the MLP layers.
        """
        return self.mlp_layers(input_feature)

    def _get_activation(self, activation: Activations = Activations.RELU):
        """Retrieve the activation to use at the end
        of a MLP layer using structural pattern matching.
        """
        match activation:
            case Activations.SIGMOID:
                return nn.Sigmoid()
            case Activations.TANH:
                return nn.Tanh()
            case Activations.RELU:
                return nn.ReLU()
            case Activations.LEAKYRELU:
                return nn.LeakyReLU()
            case _:
                raise ValueError("Activation function not supported.")

    def _init_weights(self, module: Module):
        """Initialize the weights of a module."""
        if isinstance(module, nn.Linear):
            match self.init:
                case Initializations.NORM:
                    normal_(module.weight.data, mean=0, std=0.01)
                case Initializations.XAVIER_NORM:
                    xavier_normal_(module.weight.data)
                case Initializations.XAVIER_UNI:
                    xavier_uniform_(module.weight.data)
                case _:
                    raise ValueError("Initialization mode not supported.")
            if module.bias is not None:
                module.bias.data.fill_(0.0)


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
