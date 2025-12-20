# pylint: disable = R0801, E1102, W0221
from typing import Any, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss, InfoNCELoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="XSimGCL")
class XSimGCL(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of XSimGCL algorithm from
        XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation (TKDE 2023).

    For further details, check the `paper <https://arxiv.org/abs/2209.02544>`_.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The embedding size of user and item.
        n_layers (int): The number of graph convolution layers.
        lambda_ (float): Coefficient for contrastive loss.
        eps (float): Perturbation noise scale
        temperature (float): Temperature for InfoNCE loss.
        layer_cl (int): Layer to pick for contrastive learning.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Hyperparameters
    embedding_size: int
    n_layers: int
    lambda_: float
    eps: float
    temperature: float
    layer_cl: int
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, info, *args, seed=seed, **kwargs)

        # Initialize Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        self.apply(self._init_weights)

        # Normalize Adjacency Matrix explicitly (Symmetric Normalization)
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,
            normalize=True,
        )

        # Initialize Losses
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.nce_loss = InfoNCELoss(self.temperature)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs,
    ):
        return interactions.get_pos_neg_dataloader(
            batch_size=self.batch_size, low_memory=low_memory
        )

    def _perturb_embedding(self, embedding: Tensor) -> Tensor:
        """Adds noise to embeddings: E' = E + eps * normalize(noise)."""
        noise = torch.rand_like(embedding)
        noise = F.normalize(noise, p=2, dim=1)
        return embedding + (self.eps * noise)

    def forward(
        self, perturbed: bool = False
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """Propagates embeddings through the graph.

        Args:
            perturbed (bool): If True, adds noise during propagation for CL.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]: Tuple containing:
            - user_final_emb, item_final_emb: The averaged embeddings for prediction.
            - user_cl_emb, item_cl_emb: The specific layer embeddings for CL (if perturbed=True).
        """
        ego_embeddings = self.get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )

        # Ensure adj is on the same device
        if self.adj.device() != ego_embeddings.device:
            self.adj = self.adj.to(ego_embeddings.device)

        final_embeddings = ego_embeddings.clone()
        cl_embeddings = None
        current_embeddings = ego_embeddings

        for layer_idx in range(self.n_layers):
            # Graph Convolution: E(l) = A * E(l-1)
            # SparseTensor matmul is efficient
            current_embeddings = self.adj.matmul(current_embeddings)

            # XSimGCL Perturbation
            if perturbed:
                current_embeddings = self._perturb_embedding(current_embeddings)

            # Accumulate sum for final representation
            final_embeddings.add_(current_embeddings)

            # Capture specific layer for Contrastive Learning
            if layer_idx == (self.layer_cl - 1):
                cl_embeddings = current_embeddings

        # Mean Pooling: 1/(L+1) * sum(E_0 ... E_L)
        final_embeddings.div_(self.n_layers + 1)

        # Split into user and item embeddings
        user_final, item_final = torch.split(
            final_embeddings, [self.n_users, self.n_items + 1]
        )

        user_cl, item_cl = None, None
        if cl_embeddings is not None:
            user_cl, item_cl = torch.split(
                cl_embeddings, [self.n_users, self.n_items + 1]
            )

        return user_final, item_final, user_cl, item_cl

    def train_step(self, batch: Any, *args: Any, **kwargs: Any) -> Tensor:
        user, pos_item, neg_item = batch

        # Forward Pass with Perturbation
        # We get both final embeddings (for BPR) and CL embeddings (for InfoNCE)
        users_final, items_final, users_cl, items_cl = self.forward(perturbed=True)

        # Get embeddings for current batch
        batch_users = users_final[user]
        batch_pos = items_final[pos_item]
        batch_neg = items_final[neg_item]

        # Calculate scores
        pos_scores = (batch_users * batch_pos).sum(dim=1)
        neg_scores = (batch_users * batch_neg).sum(dim=1)

        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        # Calculate loss between the final view and the specific layer view
        cl_loss_user = self.nce_loss(users_final[user], users_cl[user])
        cl_loss_item = self.nce_loss(items_final[pos_item], items_cl[pos_item])
        cl_loss = self.lambda_ * (cl_loss_user + cl_loss_item)

        # Regularize initial (ego) embeddings, not the propagated ones
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        return bpr_loss + cl_loss + reg_loss

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the learned embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Forward the propagation network without perturbation
        user_all_embeddings, item_all_embeddings, _, _ = self.forward(perturbed=False)

        # Get the embeddings for the specific users in the batch
        user_embeddings = user_all_embeddings[
            user_indices
        ]  # [batch_size, embedding_size]

        if item_indices is None:
            # Case 'full': prediction on all items
            item_embeddings = item_all_embeddings[:-1, :]  # [n_items, embedding_size]
            einsum_string = "be,ie->bi"  # b: batch, e: embedding, i: item
        else:
            # Case 'sampled': prediction on a sampled set of items
            item_embeddings = item_all_embeddings[
                item_indices
            ]  # [batch_size, pad_seq, embedding_size]
            einsum_string = "be,bse->bs"  # b: batch, e: embedding, s: sample

        predictions = torch.einsum(
            einsum_string, user_embeddings, item_embeddings
        )  # [batch_size, n_items] or [batch_size, pad_seq]
        return predictions
