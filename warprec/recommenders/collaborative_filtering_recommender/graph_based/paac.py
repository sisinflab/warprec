# pylint: disable = R0801, E1102
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss, InfoNCELoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="PAAC")
class PAAC(GraphRecommenderUtils, IterativeRecommender):
    """Implementation of PAAC from
    "Popularity-Aware Alignment and Contrast for Mitigating Popularity Bias in Recommendation" (KDD 2024).

    Popularity-Aware Alignment and Contrast (PAAC) for Mitigating Popularity Bias.

    PAAC wraps a LightGCN encoder with the supervised alignment and
    re-weighted contrastive objectives defined in the paper.

    Args:
        params (dict): Model hyperparameters (see PAACConfig).
        info (dict): Dataset metadata (n_users, n_items, …).
        interactions (Interactions): Training interactions used to build the
            graph and popularity counts.
        *args (Any): Forwarded to parent constructors.
        seed (int): Random seed for reproducibility.
        **kwargs (Any): Forwarded to parent constructors.

    Attributes:
        DATALOADER_TYPE: POS_NEG_LOADER — yields (user, pos_item, neg_item) triples.
        embedding_size (int): Embedding dimensionality.
        n_layers (int): LightGCN propagation depth.
        lambda1 (float): Weight on the supervised alignment loss (λ₁).
        lambda2 (float): Weight on the re-weighting contrastive loss (λ₂).
        temperature (float): InfoNCE temperature τ.
        gamma (float): Popular-vs-unpopular positive-sample weight γ (Eq. 7).
        beta (float): Cross-group negative-sample weight β (Eq. 8/9).
        pop_ratio (float): Fraction of batch items classified as popular per mini-batch.
        eps (float): Noise scale for contrastive augmentation.
        reg_weight (float): L2 regularization coefficient λ₃.
        batch_size (int): Training batch size.
        epochs (int): Maximum training epochs.
        learning_rate (float): Adam learning rate.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    lambda1: float
    lambda2: float
    temperature: float
    gamma: float
    beta: float
    pop_ratio: float
    eps: float
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
    ) -> None:
        super().__init__(params, info, *args, seed=seed, **kwargs)

        # ------------------------------------------------------------------ #
        # Embeddings                                                           #
        # ------------------------------------------------------------------ #
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # Normalized adjacency matrix for LightGCN propagation
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,
            normalize=True,
        )

        # ------------------------------------------------------------------ #
        # Popularity counts (Section 3.1, Eq. 3/4)                           #
        # Popularity p(i) = interaction frequency of item i in training data  #
        # ------------------------------------------------------------------ #
        # ASSUMPTION: item popularity = raw interaction count from the
        # training set, consistent with prior works cited in the paper
        # (IPS, MACR). Counts indexed by WarpRec item id (0-based).
        sparse_mat = interactions.get_sparse()  # CSR: users × items
        # Sum over users to get per-item interaction count
        item_pop = torch.tensor(
            sparse_mat.sum(axis=0).A1, dtype=torch.float32
        )  # shape [n_items]
        # Register as buffer so it moves to the correct device automatically
        self.register_buffer("item_popularity", item_pop)

        self.bpr_loss = BPRLoss()  # Eq. 1
        self.reg_loss = EmbLoss()  # λ₃ * ||Θ||²  (Eq. 11)
        self.info_nce_loss = InfoNCELoss(self.temperature)

        # Weight initialization (xavier_normal_ for Linear/Embedding)
        self.apply(self._init_weights)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ) -> DataLoader:
        return interactions.get_contrastive_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def _perturb_embedding(self, embedding: Tensor) -> Tensor:
        """Apply SimGCL-style noise perturbation."""
        noise = torch.rand_like(embedding)
        noise = noise * embedding.sign()
        noise = F.normalize(noise, p=2, dim=1) * self.eps
        return embedding + noise

    def forward(self, perturbed: bool = False) -> Tuple[Tensor, Tensor]:
        """LightGCN graph propagation with an optional perturbed view."""
        ego_embeddings = self.get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )

        # Move adjacency matrix to current device if needed
        if self.adj.device() != ego_embeddings.device:
            self.adj = self.adj.to(ego_embeddings.device)

        final_embeddings = ego_embeddings.clone()
        current_embeddings = ego_embeddings

        for _ in range(self.n_layers):
            current_embeddings = self.adj.matmul(current_embeddings)
            if perturbed:
                current_embeddings = self._perturb_embedding(current_embeddings)
            final_embeddings.add_(current_embeddings)

        all_embeddings = final_embeddings / (self.n_layers + 1)

        user_all_emb, item_all_emb = torch.split(
            all_embeddings, [self.n_users, self.n_items + 1]
        )
        return user_all_emb, item_all_emb

    def _supervised_alignment_loss(
        self,
        users: Tensor,
        pos_items: Tensor,
        item_embeddings: Tensor,
    ) -> Tensor:
        """Compute the batch-local supervised alignment loss from Eq. 5."""
        # Find unique users and items in the current batch
        unique_users, user_inverse = torch.unique(users, return_inverse=True)
        unique_items, item_inverse = torch.unique(pos_items, return_inverse=True)

        num_u = unique_users.size(0)
        num_i = unique_items.size(0)

        # Create a boolean mask of interactions in the batch (Users x Items)
        mask = torch.zeros((num_u, num_i), dtype=torch.bool, device=users.device)
        mask[user_inverse, item_inverse] = True

        # Count items per user and filter out users with fewer than 2 items
        n_items = mask.sum(dim=1)
        valid_users = n_items >= 2

        if not valid_users.any():
            return torch.tensor(0.0, device=users.device)

        # Reduce matrices to valid users only to save computation memory and time
        mask = mask[valid_users]
        n_items = n_items[valid_users]
        num_u = mask.size(0)

        # Get item popularity and calculate per-user ranks
        pops = self.item_popularity[unique_items]  # type: ignore[index]

        # Assign -inf to items the user hasn't interacted with
        # to push them to the bottom during the sorting phase
        user_item_pops = torch.where(
            mask, pops.unsqueeze(0), torch.tensor(-float("inf"), device=users.device)
        )

        # Calculate the popularity rank (0 = most popular)
        sort_idx = torch.argsort(user_item_pops, dim=1, descending=True)
        ranks = torch.empty_like(sort_idx)
        ranks.scatter_(
            1,
            sort_idx,
            torch.arange(num_i, device=users.device).unsqueeze(0).expand(num_u, num_i),
        )

        # Determine the split points (n_pop and n_unpop)
        n_pop = n_items // 2
        n_unpop = n_items - n_pop

        # Create masks for popular and unpopular items
        # An item is "pop" if the user interacted with it (mask) AND its rank is < n_pop
        pop_mask = mask & (ranks < n_pop.unsqueeze(1))
        unpop_mask = mask & (ranks >= n_pop.unsqueeze(1))

        # Extract embeddings and compute sums
        H = item_embeddings[unique_items]  # [num_i, D]
        H_sq = H.pow(2).sum(dim=1)  # [num_i]

        # Sum of squares (equivalent to sum_sq_pop / sum_sq_unpop)
        sum_sq_pop = (pop_mask.float() * H_sq.unsqueeze(0)).sum(dim=1)  # [num_u]
        sum_sq_unpop = (unpop_mask.float() * H_sq.unsqueeze(0)).sum(dim=1)  # [num_u]

        # Sum of embeddings (equivalent to sum_pop / sum_unpop)
        # We use matrix multiplication: [num_u, num_i] @ [num_i, D] -> [num_u, D]
        sum_pop = pop_mask.float() @ H  # [num_u, D]
        sum_unpop = unpop_mask.float() @ H  # [num_u, D]

        # Dot product between the sums
        dot_sums = (sum_pop * sum_unpop).sum(dim=1)  # [num_u]

        # Final loss calculation
        pair_loss = (
            n_unpop.float() * sum_sq_pop + n_pop.float() * sum_sq_unpop - 2.0 * dot_sums
        )

        loss_per_user = torch.clamp(pair_loss, min=0.0) / n_items.float()

        return loss_per_user.sum()

    def _reweighting_contrast_loss(
        self,
        item_emb_view1: Tensor,
        item_emb_view2: Tensor,
        batch_items: Tensor,
    ) -> Tensor:
        """Compute the item-side re-weighted contrastive loss."""
        if batch_items.numel() < 2:
            return item_emb_view1.new_zeros(())

        h_prime = F.normalize(item_emb_view1[batch_items], p=2, dim=1)
        h_dprime = F.normalize(item_emb_view2[batch_items], p=2, dim=1)

        pop_mask, unpop_mask = self._split_batch_items_by_popularity(batch_items)
        if not pop_mask.any() or not unpop_mask.any():
            return item_emb_view1.new_zeros(())

        return self._group_infonce(h_prime, h_dprime, pop_mask, unpop_mask)

    def _split_batch_items_by_popularity(
        self, batch_items: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Split a batch item set into top-x% popular and remaining unpopular items."""
        n_items = batch_items.numel()
        pops = self.item_popularity[batch_items]  # type: ignore[index]
        sorted_idx = torch.argsort(pops, descending=True)
        k_pop = math.ceil(self.pop_ratio * n_items)
        k_pop = min(max(k_pop, 1), n_items - 1)

        pop_mask = torch.zeros(n_items, dtype=torch.bool, device=batch_items.device)
        pop_mask[sorted_idx[:k_pop]] = True
        return pop_mask, ~pop_mask

    def _group_infonce(
        self,
        view1: Tensor,
        view2: Tensor,
        pop_mask: Tensor,
        unpop_mask: Tensor,
    ) -> Tensor:
        """Compute the item-side γ-weighted InfoNCE from Eq. 7-9."""
        log_beta = math.log(max(self.beta, 1e-8))

        # ---- L^pop (Eq. 8) --- popular items as positive samples ---------- #
        if pop_mask.sum() > 0:
            h_pop_v1 = view1[pop_mask]  # [n_pop, D]
            h_pop_v2 = view2[pop_mask]  # [n_pop, D]
            h_unpop_v2 = view2[unpop_mask]  # [n_unpop, D]

            pos_scores_pop = (h_pop_v1 * h_pop_v2).sum(dim=1) / self.temperature

            intra_pop = torch.matmul(h_pop_v1, h_pop_v2.T) / self.temperature
            log_intra_pop = torch.logsumexp(intra_pop, dim=1)

            if h_unpop_v2.numel() > 0:
                cross_pop = torch.matmul(h_pop_v1, h_unpop_v2.T) / self.temperature
                log_cross_pop = log_beta + torch.logsumexp(cross_pop, dim=1)
                denom_pop = torch.logaddexp(log_intra_pop, log_cross_pop)
            else:
                denom_pop = log_intra_pop

            l_pop = -(pos_scores_pop - denom_pop).mean()
        else:
            l_pop = torch.tensor(0.0, device=view1.device)

        # ---- L^unpop (Eq. 9) — unpopular items as positive samples ------- #
        if unpop_mask.sum() > 0:
            h_unpop_v1 = view1[unpop_mask]  # [n_unpop, D]
            h_unpop_v2 = view2[unpop_mask]  # [n_unpop, D]
            h_pop_v2_for_unpop = view2[pop_mask]  # [n_pop, D]

            pos_scores_unpop = (h_unpop_v1 * h_unpop_v2).sum(dim=1) / self.temperature

            intra_unpop = torch.matmul(h_unpop_v1, h_unpop_v2.T) / self.temperature
            log_intra_unpop = torch.logsumexp(intra_unpop, dim=1)

            if h_pop_v2_for_unpop.numel() > 0:
                cross_unpop = (
                    torch.matmul(h_unpop_v1, h_pop_v2_for_unpop.T) / self.temperature
                )
                log_cross_unpop = log_beta + torch.logsumexp(cross_unpop, dim=1)
                denom_unpop = torch.logaddexp(log_intra_unpop, log_cross_unpop)
            else:
                denom_unpop = log_intra_unpop

            l_unpop = -(pos_scores_unpop - denom_unpop).mean()
        else:
            l_unpop = torch.tensor(0.0, device=view1.device)

        return self.gamma * l_pop + (1.0 - self.gamma) * l_unpop

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Single training iteration following Algorithm 1 of the paper.

        Computes the three-component loss (Eq. 11):
            L = L_rec  +  λ₁ * L_sa  +  λ₂ * L_cl  +  λ₃ * ||Θ||²

        Args:
            batch (Any): Tuple of (user, pos_item, neg_item) from POS_NEG_LOADER.
            batch_idx (int): Batch index (unused).

        Returns:
            Tensor: Scalar total loss.
        """
        user, pos_item, neg_item = batch

        user_emb, item_emb = self.forward(perturbed=False)
        user_emb_v1, item_emb_v1 = self.forward(perturbed=True)
        user_emb_v2, item_emb_v2 = self.forward(perturbed=True)

        u_emb = user_emb[user]
        pos_emb = item_emb[pos_item]
        neg_emb = item_emb[neg_item]

        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
        rec_loss = self.bpr_loss(pos_scores, neg_scores)

        sa_loss = self._supervised_alignment_loss(user, pos_item, item_emb)

        batch_items = pos_item.unique()
        batch_users = user.unique()
        item_cl_loss = self._reweighting_contrast_loss(
            item_emb_v1, item_emb_v2, batch_items
        )
        user_cl_loss = self.info_nce_loss(
            user_emb_v1[batch_users], user_emb_v2[batch_users]
        )
        cl_loss = 0.5 * (item_cl_loss + user_cl_loss)

        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        loss = rec_loss + self.lambda1 * sa_loss + self.lambda2 * cl_loss + reg_loss

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("rec_loss", rec_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("sa_loss", sa_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("cl_loss", cl_loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Score users against items using the learnt propagated embeddings.

        Prediction score: s(u, i) = z_u^T h_i  (dot product, Section 2.1)

        Args:
            user_indices (Tensor): Batch of user indices [B].
            *args (Any): Ignored.
            item_indices (Optional[Tensor]): If None, scores all items [B, N].
                Otherwise scores the provided candidates [B, K].
            **kwargs (Any): Ignored.

        Returns:
            Tensor: Score matrix of shape [B, N] (full) or [B, K] (sampled).
        """
        user_all_emb, item_all_emb = self.propagate_embeddings()
        user_emb = user_all_emb[user_indices]  # [B, D]

        if item_indices is None:
            # Full prediction over all items (exclude padding index)
            item_emb = item_all_emb[:-1, :]  # [N, D]
            return torch.einsum("be,ie->bi", user_emb, item_emb)
        else:
            # Sampled prediction
            item_emb = item_all_emb[item_indices]  # [B, K, D]
            return torch.einsum("be,bke->bk", user_emb, item_emb)
