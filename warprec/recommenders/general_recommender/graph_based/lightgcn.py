# pylint: disable = R0801, E1102
from typing import Optional, Callable, Tuple, Any

import torch
import torch_geometric
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.init import xavier_normal_
from torch_geometric.nn import LGConv
from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix

from warprec.data.dataset import Interactions
from warprec.recommenders.base_recommender import Recommender, GraphRecommenderUtils
from warprec.recommenders.losses import BPRLoss, EmbeddingLoss
from warprec.utils.registry import model_registry


@model_registry.register(name="LightGCN")
class LightGCN(Recommender, GraphRecommenderUtils):
    """Implementation of LightGCN algorithm from
        LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR 2020)

    For further details, check the `paper <https://arxiv.org/abs/2002.02126>`_.

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
        n_layers (int): The number of graph convolution layers.
        reg_weight (float): The weight decay for L2 regularization.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    reg_weight: float
    epochs: int
    learning_rate: float

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
        self._name = "LightGCN"

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

        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # Init embedding weights
        self.apply(self._init_weights)

        # Loss and optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbeddingLoss(norm=2)  # L2-norm for embeddings

        # Adjacency tensor initialization
        self.adj: Optional[SparseTensor] = None

        # Initialization of the propagation network
        propagation_network_list = []
        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(), "x, edge_index -> x"))
        self.propagation_network = torch_geometric.nn.Sequential(
            "x, edge_index", propagation_network_list
        )

        # Normalization for embedding normalization
        self.alpha = torch.tensor(
            [1 / (k + 1) for k in range(self.n_layers + 1)], device=self._device
        )

        # Move to device
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
        """Main train method for LightGCN.

        Args:
            interactions (Interactions): The interactions that will be used to train the model.
            *args (Any): List of arguments.
            report_fn (Optional[Callable]): The Ray Tune function to report the iteration.
            **kwargs (Any): The dictionary of keyword arguments.
        """
        if self.adj is None:
            self.adj = self._get_norm_adj_mat(
                interactions.get_sparse().tocoo(),
                self.n_users,
                self.n_items,
                self._device,
            )

        # Get the dataloader from interactions for pairwise training
        dataloader = interactions.get_pos_neg_dataloader()

        # Training loop
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
                mf_loss: Tensor = self.mf_loss(pos_scores, neg_scores)

                # Calculate embedding regularization loss
                reg_loss = self.reg_loss(
                    self.user_embedding,
                    self.item_embedding,
                )

                loss: Tensor = mf_loss + self.reg_weight * reg_loss

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if report_fn is not None:
                report_fn(self, loss=epoch_loss)

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Forward pass of the LightGCN model for embedding propagation.

        Returns:
            Tuple[Tensor, Tensor]: User and item embeddings after propagation.
        """
        ego_embeddings = self._get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )
        embeddings_list = [ego_embeddings]

        # This will handle the propagation layer by layer.
        # This is used later to correctly multiply each layer by
        # the corresponding value of alpha
        current_embeddings = ego_embeddings
        for layer_module in self.propagation_network.children():
            current_embeddings = layer_module(current_embeddings, self.adj)
            embeddings_list.append(current_embeddings)

        # Aggregate embeddings using the alpha value
        lightgcn_all_embeddings = torch.zeros_like(ego_embeddings, device=self._device)
        for k in range(len(embeddings_list)):
            lightgcn_all_embeddings += embeddings_list[k] * self.alpha[k]

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    @torch.no_grad()
    def predict(
        self, interaction_matrix: csr_matrix, *args: Any, **kwargs: Any
    ) -> Tensor:
        """Prediction using the learned embeddings."""
        if self.adj is None:
            train_set: csr_matrix = kwargs.get("train_set", None)
            if train_set is None:
                raise RuntimeError(
                    "The model has not been fit yet. Call fit() before predict()."
                )
            self.adj = self._get_norm_adj_mat(
                train_set.tocoo(), self.n_users, self.n_items, self._device
            )

        # Perform forward pass to get the final embeddings
        user_e, item_e = self.forward()

        start_idx = kwargs.get("start", 0)
        end_idx = kwargs.get("end", interaction_matrix.shape[0])
        block_size = 200  # Better memory management

        preds = []
        # Process users in batches to manage memory
        for current_batch_start in range(start_idx, end_idx, block_size):
            current_batch_end = min(current_batch_start + block_size, end_idx)
            users_in_batch = torch.arange(
                current_batch_start, current_batch_end, device=self._device
            )

            # Get user embeddings for the current batch
            u_embeddings = user_e[users_in_batch]

            # Calculate scores by dot product with all item embeddings
            # This is the "full_sort_predict" logic
            scores_batch = torch.matmul(u_embeddings, item_e.transpose(0, 1))
            preds.append(scores_batch)

        predictions = torch.cat(preds, dim=0)  # [batch_size x num_items]

        coo = interaction_matrix.tocoo()
        user_indices_in_batch = torch.from_numpy(coo.row).to(self._device)
        item_indices = torch.from_numpy(coo.col).to(self._device)
        predictions[user_indices_in_batch, item_indices] = -torch.inf

        return predictions
