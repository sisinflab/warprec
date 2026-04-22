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


@model_registry.register(name="SimGCL")
class SimGCL(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of SimGCL from
        "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning
        for Recommendation" (SIGIR 2022).

    SimGCL discards graph augmentations entirely and instead adds uniform
    random noise to node embeddings at each GCN layer to create contrastive
    views.  Two independently perturbed views are generated per forward pass;
    an InfoNCE loss maximizes agreement between same-node representations
    across the two views while a BPR loss drives the recommendation task.

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
        n_layers (int): The number of graph convolution layers.
        lambda_ (float): Coefficient for the contrastive loss.
        eps (float): L2 norm of the perturbation noise vectors.
        temperature (float): Temperature for InfoNCE loss.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    lambda_: float  # Eq. 1 — weight of the CL loss in the joint objective
    eps: float  # Eq. 7 — perturbation magnitude (||Δ||_2 = ε)
    temperature: float  # Eq. 2 — InfoNCE temperature τ
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

        # Initialize Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        self.apply(self._init_weights)

        # Symmetric-normalized adjacency matrix  — Eq. 3
        # Only built once; SimGCL never reconstructs/perturbs the graph.
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,
            normalize=True,
        )

        # Losses
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.nce_loss = InfoNCELoss(self.temperature)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_contrastive_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def _perturb_embedding(self, embedding: Tensor) -> Tensor:
        """Add noise to embeddings following Eq. 7 of the paper.

        Noise generation:
            Δ_bar ~ U(0, 1)  (same shape as embedding)
            Δ = Δ_bar ⊙ sign(e_i)        — same hyperoctant constraint
            Δ = eps * Δ / ||Δ||_2         — normalize to L2 = eps

        Args:
            embedding (Tensor): Node embeddings [N, d].

        Returns:
            Tensor: Perturbed embeddings [N, d].
        """
        # Eq. 7 — sample uniform noise and constrain to same hyperoctant
        noise = torch.rand_like(embedding)  # Δ_bar ~ U(0,1)
        noise = noise * embedding.sign()  # Δ = Δ_bar ⊙ sign(e_i)
        # Normalize each row to unit L2 norm, then scale by eps
        noise = F.normalize(noise, p=2, dim=1) * self.eps  # ||Δ||_2 = eps
        return embedding + noise

    def forward(
        self, perturbed: bool = False
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """Propagate embeddings through the LightGCN encoder.

        When ``perturbed=True`` two independent noise-augmented views are
        produced alongside the clean final embeddings.

        # SIMPLIFICATION: The paper states that E(0) (the raw ego embeddings)
        # is skipped in the final aggregation (Eq. 8).  We follow this exactly:
        # E = (1/L) * sum_{l=1}^{L} E^(l).

        Args:
            perturbed (bool): If True, generates two noisy contrastive views.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
                - user_final, item_final: clean aggregated embeddings
                - user_cl, item_cl: None when perturbed=False; otherwise a tuple
                  of (view1_users, view1_items, view2_users, view2_items) packed
                  as two extra return values would break the interface, so we
                  return them via the instance attributes ``_cl_view1`` and
                  ``_cl_view2`` instead.
        """
        ego_embeddings = self.get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )

        # Ensure adj is on the same device
        if self.adj.device() != ego_embeddings.device:
            self.adj = self.adj.to(ego_embeddings.device)

        # --- Clean encoder (used for the BPR recommendation loss) ---
        # Eq. 8 — skip E(0): aggregate only layers 1 .. L
        clean_sum = torch.zeros_like(ego_embeddings)
        current = ego_embeddings
        for _ in range(self.n_layers):
            current = self.adj.matmul(current)  # E^(l) = A * E^(l-1)
            clean_sum.add_(current)
        # Mean pooling: (1/L) * sum
        clean_final = clean_sum / self.n_layers  # Eq. 8

        user_final, item_final = torch.split(
            clean_final, [self.n_users, self.n_items + 1]
        )

        # --- Perturbed encoders (two independent views for CL) ---
        if perturbed:
            # View 1 — Eq. 8 with noise at every layer
            v1_sum = torch.zeros_like(ego_embeddings)
            v1_current = ego_embeddings
            for _ in range(self.n_layers):
                v1_current = self.adj.matmul(v1_current)
                v1_current = self._perturb_embedding(v1_current)  # Eq. 7
                v1_sum.add_(v1_current)
            v1_final = v1_sum / self.n_layers

            # View 2 — Eq. 8 with independent noise at every layer
            v2_sum = torch.zeros_like(ego_embeddings)
            v2_current = ego_embeddings
            for _ in range(self.n_layers):
                v2_current = self.adj.matmul(v2_current)
                v2_current = self._perturb_embedding(v2_current)  # Eq. 7
                v2_sum.add_(v2_current)
            v2_final = v2_sum / self.n_layers

            user_v1, item_v1 = torch.split(v1_final, [self.n_users, self.n_items + 1])
            user_v2, item_v2 = torch.split(v2_final, [self.n_users, self.n_items + 1])
            # Store views for training_step to consume
            self._cl_view1 = (user_v1, item_v1)
            self._cl_view2 = (user_v2, item_v2)

        return user_final, item_final, None, None

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Compute the joint BPR + CL + L2 loss.

        Loss = L_rec + lambda * L_cl + reg_weight * L_reg    (Eq. 1)

        Args:
            batch (Any): Tuple of (user, pos_item, neg_item).
            batch_idx (int): The current batch index.

        Returns:
            Tensor: The computed loss for the batch.
        """
        user, pos_item, neg_item = batch

        # Forward pass: clean embeddings + two perturbed views
        users_final, items_final, _, _ = self.forward(perturbed=True)

        # Retrieve the two CL views generated during forward
        user_v1, item_v1 = self._cl_view1
        user_v2, item_v2 = self._cl_view2

        # --- BPR recommendation loss — Eq. 5 ---
        batch_users = users_final[user]
        batch_pos = items_final[pos_item]
        batch_neg = items_final[neg_item]

        pos_scores = (batch_users * batch_pos).sum(dim=1)
        neg_scores = (batch_users * batch_neg).sum(dim=1)
        bpr_loss = self.bpr_loss(pos_scores, neg_scores)

        # --- Contrastive loss — Eq. 2 (InfoNCE between view1 and view2) ---
        cl_loss_user = self.nce_loss(user_v1[user], user_v2[user])
        cl_loss_item = self.nce_loss(item_v1[pos_item], item_v2[pos_item])
        cl_loss = self.lambda_ * (cl_loss_user + cl_loss_item)  # Eq. 1

        # --- L2 regularization on ego embeddings ---
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        # Joint loss — Eq. 1
        loss = bpr_loss + cl_loss + reg_loss
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

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
        # ASSUMPTION: At inference time we use the clean encoder (no noise).
        # This is consistent with the paper which only uses noise during training.
        user_all_embeddings, item_all_embeddings = self.propagate_embeddings()

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
