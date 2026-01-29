# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="LightGODE")
class LightGODE(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of LightGODE from "Do We Really Need Graph Convolution During Training?
    Light Post-Training Graph-ODE for Efficient Recommendation" (CIKM '24).

    LightGODE skips graph convolution during training, optimizing embeddings directly
    via Alignment and Uniformity losses (like Matrix Factorization but with geometric losses).
    During inference, it applies a continuous Graph-ODE solver to inject high-order
    connectivity information.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The embedding size of user and item.
        gamma (float): The weight for the uniformity loss.
        t (float): The time horizon for ODE integration.
        n_ode_steps (int): The number of ODE integration steps during inference.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    DATALOADER_TYPE = DataLoaderType.ITEM_RATING_LOADER

    # Hyperparameters
    embedding_size: int
    gamma: float
    t: float
    n_ode_steps: int
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

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

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # Graph Construction (Used ONLY for Post-Training ODE)
        # Standard LightGCN normalization: D^-1/2 A D^-1/2
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,
            normalize=True,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Regularization
        self.reg_loss = EmbLoss()

        # Cache for inference embeddings
        self._cached_inference_emb: Optional[Tuple[Tensor, Tensor]] = None

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        low_memory: bool = False,
        **kwargs: Any,
    ):
        return interactions.get_item_rating_dataloader(
            neg_samples=0, batch_size=self.batch_size, low_memory=low_memory
        )

    def train(self, mode: bool = True):
        """Override train mode to clear cache when switching back to training."""
        super().train(mode)
        if mode:
            self._cached_inference_emb = None
        return self

    def _alignment_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Calculates Alignment Loss.

        Minimizes Euclidean distance between normalized positive pairs.
        L_align = Mean( ||x - y||^2 )
        """
        # Note: Inputs x, y are already normalized in train_step
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def _uniformity_loss(self, x: Tensor) -> Tensor:
        """Calculates Uniformity Loss.

        Maximizes distance between all pairs in the batch (Gaussian potential).
        L_uniform = log( Mean( exp( -2 * ||u - u'||^2 ) ) )
        """
        # x @ x.T gives cosine similarity matrix since x is normalized
        sq_dist = torch.mm(x, x.t())  # Cosine similarity

        # Efficient computation:
        # ||u - u'||^2 = ||u||^2 + ||u'||^2 - 2 u.u' = 2 - 2 u.u' (since norm is 1)
        # exponent = -2 * (2 - 2 * sim) = -4 + 4 * sim

        exponent = -4.0 + 4.0 * sq_dist
        return torch.logsumexp(exponent, dim=1).mean()

    @torch.no_grad()
    def _ode_solver(self) -> Tuple[Tensor, Tensor]:
        """Post-Training Graph-ODE Solver.

        Solves: dh(t)/dt = Adj * h(t) + h0
        Using Euler method: h_{k+1} = h_k + step_size * (Adj * h_k + h0)
        """
        h0_u = self.user_embedding.weight
        h0_i = self.item_embedding.weight
        h0_all = torch.cat([h0_u, h0_i], dim=0)

        h_t = h0_all.clone()

        # Ensure adj is on device
        if self.adj.device() != self.device:
            self.adj = self.adj.to(self.device)

        # Euler Integration
        # Total time T, split into N steps. Step size = T / N
        step_size = self.t / self.n_ode_steps

        for _ in range(self.n_ode_steps):
            # derivative = A * h(t) + h0

            # Graph Aggregation: A * h(t)
            agg = self.adj.matmul(h_t)

            # Add Source Term: + h0
            derivative = agg + h0_all

            # Euler Update: h(t+1) = h(t) + dt * derivative
            h_t = h_t + step_size * derivative

        user_final, item_final = torch.split(h_t, [self.n_users, self.n_items + 1])
        return user_final, item_final

    def train_step(self, batch: Any, *args, **kwargs) -> Tensor:
        user, pos_item, _ = batch  # Ignore rating values

        # Get Embeddings (No GCN)
        all_users, all_items = self.forward()

        u_emb = all_users[user]
        i_emb = all_items[pos_item]

        # Normalize (Crucial for Alignment/Uniformity)
        u_emb_norm = F.normalize(u_emb, dim=1)
        i_emb_norm = F.normalize(i_emb, dim=1)

        # Calculate Alignment loss
        align_loss = self._alignment_loss(u_emb_norm, i_emb_norm)

        # Calculate Uniformity loss
        unif_loss_u = self._uniformity_loss(u_emb_norm)
        unif_loss_i = self._uniformity_loss(i_emb_norm)
        unif_loss = self.gamma * (unif_loss_u + unif_loss_i) / 2

        # Calculate L2 loss
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user), self.item_embedding(pos_item)
        )

        return align_loss + unif_loss + reg_loss

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Standard forward pass.

        - If training: Returns raw ID embeddings (no convolution).
        - If eval: Returns ODE-convolved embeddings (with caching).
        """
        if self.training:
            # Training Phase: No Graph Convolution
            return self.user_embedding.weight, self.item_embedding.weight

        # Inference Phase: Post-Training Graph-ODE
        if self._cached_inference_emb is None:
            self._cached_inference_emb = self._ode_solver()
        return self._cached_inference_emb

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
        # Retrieve all user and item embeddings from the propagation network
        user_all_embeddings, item_all_embeddings = self.forward()

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
