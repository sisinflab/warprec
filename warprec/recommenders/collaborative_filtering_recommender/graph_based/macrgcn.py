# pylint: disable = R0801, E1102
from typing import Any, Optional, Tuple

import torch
import torch_geometric
from torch import nn, Tensor
from torch_geometric.nn import LGConv

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="MACRGCN")
class MACRGCN(GraphRecommenderUtils, IterativeRecommender):
    """Implementation of MACRGCN
        from Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System (KDD 2021).

    The model adds two auxiliary branches (user module, item module) to a
    standard LightGCN backbone and applies counterfactual inference at test
    time to remove the direct effect of item popularity on ranking scores.

    Args:
        params (dict): Model parameters.
        info (dict): Dataset information (n_users, n_items, ...).
        interactions (Interactions): Training interactions for adjacency matrix.
        *args (Any): Variable length argument list.
        seed (int): Random seed.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: Uses pointwise loader with negatives for BCE training.
        embedding_size (int): Embedding dimension.
        n_layers (int): Number of LightGCN propagation layers.
        reg_weight (float): L2 regularization weight.
        alpha (float): Weight for item module loss L_I.
        beta (float): Weight for user module loss L_U.
        c (float): Counterfactual reference constant.
        user_mlp_hidden (int): Hidden size for user module MLP.
        item_mlp_hidden (int): Hidden size for item module MLP.
        neg_samples (int): Negative samples per positive.
        batch_size (int): Batch size.
        epochs (int): Training epochs.
        learning_rate (float): Learning rate.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.ITEM_RATING_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    reg_weight: float
    alpha: float
    beta: float
    c: float
    user_mlp_hidden: int
    item_mlp_hidden: int
    neg_samples: int
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

        # ======================== Backbone: LightGCN ========================
        # Eq. 7 main branch — user-item matching y_k = Y_k(K(U=u, I=i))
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # Adjacency matrix (symmetric normalization handled by LGConv)
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,  # +1 for padding index
        )

        # LightGCN propagation layers — Section 3.2, graph convolution for K(U,I)
        propagation_network_list = []
        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(), "x, edge_index -> x"))
        self.propagation_network = torch_geometric.nn.Sequential(
            "x, edge_index", propagation_network_list
        )

        # ======================== User Module ========================
        # Section 3.3, Figure 5 (blue branch)
        # y_u = Y_u(U = u): projects user embedding to scalar score
        # ASSUMPTION: 2-layer MLP with ReLU. The paper states "can be
        # implemented as multi-layer perceptrons" but does not specify depth.
        self.user_module = nn.Sequential(
            nn.Linear(self.embedding_size, self.user_mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.user_mlp_hidden, 1),
        )

        # ======================== Item Module ========================
        # Section 3.3, Figure 5 (green branch)
        # y_i = Y_i(I = i): projects item embedding to scalar score
        # ASSUMPTION: Same 2-layer MLP architecture as user module.
        self.item_module = nn.Sequential(
            nn.Linear(self.embedding_size, self.item_mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.item_mlp_hidden, 1),
        )

        # ======================== Loss functions ========================
        # Eq. 6, 8 — BCE loss for L_O, L_I, L_U
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.reg_loss = EmbLoss()

        # Weight initialization — Appendix B: Xavier (matches _init_weights)
        self.apply(self._init_weights)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_pointwise_dataloader(
            neg_samples=self.neg_samples,
            batch_size=self.batch_size,
            **kwargs,
        )

    def forward(self) -> Tuple[Tensor, Tensor]:
        """LightGCN forward pass — propagate and average embeddings.

        Returns:
            Tuple[Tensor, Tensor]: (user_embeddings, item_embeddings) after
                multi-layer graph convolution with mean pooling across layers.
        """
        # Section 3.2 — K(U, I) via LightGCN graph convolution
        ego_embeddings = self.get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )

        if self.adj.device() != ego_embeddings.device:
            self.adj = self.adj.to(ego_embeddings.device)

        embeddings_list = [ego_embeddings]

        current_embeddings = ego_embeddings
        for layer_module in self.propagation_network.children():
            current_embeddings = layer_module(current_embeddings, self.adj)
            embeddings_list.append(current_embeddings)

        # LightGCN layer combination: mean pooling (equivalent to uniform alpha)
        stacked_embeddings = torch.stack(embeddings_list, dim=0)
        lightgcn_all_embeddings = torch.mean(stacked_embeddings, dim=0)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings,
            [self.n_users, self.n_items + 1],
        )
        return user_all_embeddings, item_all_embeddings

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Multi-task training with BCE losses on three branches.

        Implements Eq. 7 (fusion) and Eq. 8 (multi-task loss):
            L = L_O + alpha * L_I + beta * L_U

        Args:
            batch (Any): Tuple of (user, item, rating) from pointwise loader.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Combined loss scalar.
        """
        user, item, rating = batch[:3]
        rating = rating.float()

        # --- Backbone: LightGCN propagated embeddings ---
        user_all_embeddings, item_all_embeddings = self.forward()

        u_emb = user_all_embeddings[user]  # [batch, emb_size]
        i_emb = item_all_embeddings[item]  # [batch, emb_size]

        # y_k: user-item matching score (dot product)
        # Section 3.3 — "ranking score from the existing recommender"
        y_k = (u_emb * i_emb).sum(dim=-1)  # [batch]

        # y_u: user conformity score — Section 3.3, user module
        # Uses the *initial* (ego) user embedding as input to the user module,
        # consistent with the causal graph U -> Y (direct effect from user node).
        # ASSUMPTION: Use ego embedding (pre-propagation) for user/item modules,
        # since the causal graph treats U and I as raw inputs, not propagated ones.
        u_ego = self.user_embedding(user)  # [batch, emb_size]
        y_u = self.user_module(u_ego).squeeze(-1)  # [batch]

        # y_i: item popularity score — Section 3.3, item module
        i_ego = self.item_embedding(item)  # [batch, emb_size]
        y_i = self.item_module(i_ego).squeeze(-1)  # [batch]

        # --- Eq. 7: fused ranking score y_ui = y_k * sigma(y_i) * sigma(y_u) ---
        y_ui = y_k * torch.sigmoid(y_i) * torch.sigmoid(y_u)

        # --- Eq. 8: multi-task loss ---
        # L_O: main recommendation loss on fused score
        loss_o = self.bce_loss(y_ui, rating)  # Eq. 6

        # L_I: item module loss — trains item module to predict interaction
        # from item alone (captures popularity)
        loss_i = self.bce_loss(y_i, rating)

        # L_U: user module loss — trains user module to predict interaction
        # from user alone (captures conformity)
        loss_u = self.bce_loss(y_u, rating)

        # L2 regularization on ego embeddings
        reg = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(item),
        )

        # Eq. 8: L = L_O + alpha * L_I + beta * L_U + reg
        loss = loss_o + self.alpha * loss_i + self.beta * loss_u + reg

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("loss_o", loss_o, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss_i", loss_i, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss_u", loss_u, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Counterfactual inference — Eq. 9, Algorithm 1.

        At test time the ranking score is:
            y_ui = y_k * sigma(y_i) * sigma(y_u)  -  c * sigma(y_i) * sigma(y_u)
                 = (y_k - c) * sigma(y_i) * sigma(y_u)

        This removes the Natural Direct Effect (NDE) of item popularity
        (Section 3.4, Eq. 10), ranking items by Total Indirect Effect (TIE).

        Args:
            user_indices (Tensor): Batch of user indices.
            *args (Any): Variable length argument list.
            item_indices (Optional[Tensor]): Candidate item indices. If None,
                scores are computed for all items.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Tensor: Debiased ranking scores [batch_size, n_items] or
                    [batch_size, k] if item_indices provided.
        """
        # Propagated embeddings (cached during eval via GraphRecommenderUtils)
        user_all_embeddings, item_all_embeddings = self.propagate_embeddings()

        user_emb = user_all_embeddings[user_indices]  # [batch, emb_size]

        # --- User module score ---
        u_ego = self.user_embedding(user_indices)  # [batch, emb_size]
        y_u = self.user_module(u_ego).squeeze(-1)  # [batch]
        sigma_y_u = torch.sigmoid(y_u)  # [batch]

        if item_indices is None:
            # Full prediction: score all items (excluding padding)
            item_emb = item_all_embeddings[:-1, :]  # [n_items, emb_size]

            # y_k for all items: [batch, n_items]
            y_k = torch.matmul(user_emb, item_emb.t())

            # Item module scores for all items
            i_ego = self.item_embedding.weight[:-1, :]  # [n_items, emb_size]
            y_i = self.item_module(i_ego).squeeze(-1)  # [n_items]
            sigma_y_i = torch.sigmoid(y_i)  # [n_items]

            # Eq. 9: counterfactual debiased score
            # (y_k - c) * sigma(y_i)[1, n_items] * sigma(y_u)[batch, 1]
            scores = (y_k - self.c) * sigma_y_i.unsqueeze(0) * sigma_y_u.unsqueeze(1)
        else:
            # Sampled prediction
            item_emb = item_all_embeddings[item_indices]  # [batch, k, emb_size]

            # y_k: [batch, k]
            y_k = torch.einsum("be,bke->bk", user_emb, item_emb)

            # Item module scores for sampled items
            i_ego = self.item_embedding(item_indices)  # [batch, k, emb_size]
            y_i = self.item_module(i_ego).squeeze(-1)  # [batch, k]
            sigma_y_i = torch.sigmoid(y_i)  # [batch, k]

            # Eq. 9: counterfactual debiased score
            scores = (y_k - self.c) * sigma_y_i * sigma_y_u.unsqueeze(1)

        return scores
