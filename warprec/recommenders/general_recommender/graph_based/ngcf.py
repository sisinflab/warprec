# pylint: disable = R0801, E1102
from typing import Optional, Callable, Tuple, Any

import torch
import numpy as np
import scipy.sparse as sp
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_normal_
from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix, coo_matrix

from warprec.data.dataset import Interactions
from warprec.recommenders.base_recommender import Recommender, GraphRecommenderUtils
from warprec.recommenders.losses import BPRLoss
from warprec.recommenders.layers import SparseDropout, NGCFLayer
from warprec.utils.registry import model_registry


@model_registry.register(name="NGCF")
class NGCF(Recommender, GraphRecommenderUtils):
    """Implementation of NGCF algorithm from
        Neural Graph Collaborative Filtering (SIGIR 2019)

    For further details, check the `paper <https://arxiv.org/abs/1905.08166>`_.

    Args:
        params (dict): Model parameters.
        *args (Any): Variable length argument list.
        device (str): The device used for tensor operations.
        seed (int): The seed to use for reproducibility.
        info (dict): The dictionary containing dataset information.
        **kwargs (Any): Arbitrary keyword arguments.

    Raises:
        ValueError: If the items or users value was not passed through the info dict.

    Attributes:
        embedding_size (int): The embedding size of user and item.
        weight_decay (float): The value of weight decay used in the optimizer.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        weight_size (list[int]): List of hidden sizes for each layer.
        node_dropout (float): Dropout rate for nodes in the adjacency matrix.
        message_dropout (float): Dropout rate for messages/embeddings during propagation.
    """

    # Model hyperparameters
    embedding_size: int
    weight_decay: float
    epochs: int
    learning_rate: float
    weight_size: list[int]
    node_dropout: float
    message_dropout: float

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
        self._name = "NGCF"

        # Get information from dataset info
        self.n_users = info.get("users", None)
        if not self.n_users:
            raise ValueError(
                "Users value must be provided to correctly initialize the model."
            )
        self.n_items = info.get("items", None)
        if not self.n_items:
            raise ValueError(
                "Items value must be provided to correctly initialize the model."
            )
        self.block_size = kwargs.get("block_size", 50)

        # Initialize the hidden dimensions
        self.hidden_size_list = [
            self.embedding_size
        ] + self.weight_size  # [embed_k, layer1_dim, layer2_dim, ...]

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Init embedding weights
        self.apply(self._init_weights)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.loss = BPRLoss()
        self.adj_matrix: Optional[SparseTensor] = None

        # Optionally define a dropout layer (optimized for sparse data)
        self.sparse_dropout = (
            SparseDropout(self.node_dropout) if self.node_dropout > 0 else None
        )

        # Initialization of the propagation network
        self.propagation_network = nn.ModuleList()
        for i in range(len(self.weight_size)):
            in_f = self.hidden_size_list[i]
            out_f = self.hidden_size_list[i + 1]
            self.propagation_network.append(
                NGCFLayer(in_f, out_f, self.message_dropout)
            )

        self.to(self._device)

    def _init_weights(self, module: Module):
        """Internal method to initialize weights.

        Args:
            module (Module): The module to initialize.
        """
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)

    def fit(
        self,
        interactions: Interactions,
        *args: Any,
        report_fn: Optional[Callable] = None,
        **kwargs: Any,
    ):
        """Main train method for NGCF.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        # Compute the normalized adjacency matrix
        if self.adj_matrix is None:
            self.adj_matrix = self._get_norm_adj_mat_ngcf(
                interactions.get_sparse().tocoo(),
                self.n_users,
                self.n_items,
                self._device,
            )

        # Get the dataloader from interactions for pairwise training
        dataloader = interactions.get_pos_neg_dataloader()

        self.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                user, pos_item, neg_item = [x.to(self._device) for x in batch]

                self.optimizer.zero_grad()

                # Get propagated embeddings
                user_all_embeddings, item_all_embeddings = self.forward()

                # Get embeddings for current batch users and items
                u_embeddings = user_all_embeddings[user]
                pos_embeddings = item_all_embeddings[pos_item]
                neg_embeddings = item_all_embeddings[neg_item]

                # Calculate BPR Loss
                pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
                neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
                loss: Tensor = self.loss(pos_scores, neg_scores)

                # Loss computation and backpropagation
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if report_fn is not None:
                report_fn(self, loss=epoch_loss)

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Forward pass of the NGCF model for embedding propagation.

        Returns:
            Tuple[Tensor, Tensor]: User and item embeddings after propagation.
        """
        # Get the ego_embeddings [user + item]
        ego_embeddings = self._get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )
        embeddings_list = [ego_embeddings]

        # Apply dropout if required from hyperparameters
        adj_matrix_current = self.adj_matrix
        if self.sparse_dropout is not None:
            adj_matrix_current = self.sparse_dropout(self.adj_matrix)

        # Forward each embedding through the sequential
        # propagation network
        current_embeddings = ego_embeddings
        for layer_module in self.propagation_network:
            current_embeddings = layer_module(current_embeddings, adj_matrix_current)
            embeddings_list.append(current_embeddings)

        # Concatenate embeddings from all layers (including ego-embeddings)
        # along the feature dimension
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction using the learned embeddings."""
        if self.adj_matrix is None:
            train_set: csr_matrix = kwargs.get("train_set", None)
            if train_set is None:
                raise RuntimeError(
                    "The model has not been fit yet. Call fit() before predict()."
                )
            self.adj_matrix = self._get_norm_adj_mat_ngcf(
                train_set.tocoo(), self.n_users, self.n_items, self._device
            )

        # Perform forward pass to get the final embeddings
        user_e, item_e = self.forward()

        start_idx = kwargs.get("start", 0)
        end_idx = kwargs.get("end", interaction_matrix.shape[0])

        preds = []
        # Process users in batches to manage memory
        for current_batch_start in range(start_idx, end_idx, self.block_size):
            current_batch_end = min(current_batch_start + self.block_size, end_idx)
            users_in_batch = torch.arange(
                current_batch_start, current_batch_end, device=self._device
            )

            # Get user embeddings for the current batch
            u_embeddings = user_e[users_in_batch]

            # Calculate scores by dot product with all item embeddings
            scores_batch = torch.matmul(u_embeddings, item_e.transpose(0, 1))
            preds.append(scores_batch)

        predictions = torch.cat(preds, dim=0)  # [batch_size x num_items]

        coo = interaction_matrix.tocoo()
        user_indices_in_batch = torch.from_numpy(coo.row).to(self._device)
        item_indices = torch.from_numpy(coo.col).to(self._device)
        predictions[user_indices_in_batch, item_indices] = -torch.inf

        return predictions

    def _get_norm_adj_mat_ngcf(
        self,
        interaction_matrix: coo_matrix,
        n_users: int,
        n_items: int,
        device: torch.device | str = "cpu",
    ) -> SparseTensor:
        """Get the normalized interaction matrix of users and items specific to NGCF.
        This includes constructing the full adjacency matrix and applying symmetric normalization.

        Args:
            interaction_matrix (coo_matrix): The full interaction matrix in coo format.
            n_users (int): The number of users.
            n_items (int): The number of items.
            device (torch.device | str): Device to use for the adjacency matrix.

        Returns:
            SparseTensor: The sparse normalized adjacency matrix (A_hat).
        """
        # Build adjacency matrix (A)
        # [num_user + num_items x num_user + num_items]
        A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()

        # Add user-item interactions
        for u, i in zip(inter_M.row, inter_M.col):
            A[u, i + n_users] = 1.0  # user -> item
        # Add item-user interactions (transpose)
        for i, u in zip(inter_M_t.row, inter_M_t.col):
            A[i + n_users, u] = 1.0  # item -> user

        A = (
            A.tocsr()
        )  # Convert to CSR for efficient row-wise sum and diagonal matrix creation

        # Symmetric Normalization: D^{-0.5} A D^{-0.5}
        sum_rows = np.array(A.sum(axis=1)).flatten()
        # Add epsilon to avoid division by zero
        sum_rows[sum_rows == 0] = 1e-7
        diag_inv_sqrt = np.power(sum_rows, -0.5)
        D_inv_sqrt = sp.diags(diag_inv_sqrt)

        # L = D^{-0.5} A D^{-0.5}
        L = D_inv_sqrt.dot(A).dot(D_inv_sqrt)

        # Convert to COO format for SparseTensor
        L_coo = L.tocoo()
        indices = torch.LongTensor(np.vstack((L_coo.row, L_coo.col)))
        values = torch.FloatTensor(L_coo.data)
        shape = torch.Size(L_coo.shape)

        return torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)
