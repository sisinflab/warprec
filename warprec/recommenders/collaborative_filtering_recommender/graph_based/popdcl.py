# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional

import torch
import numpy as np
import torch_geometric
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.nn import LGConv

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="PopDCL")
class PopDCL(GraphRecommenderUtils, IterativeRecommender):
    """Implementation of PopDCL model from
        Popularity-aware Debiased Contrastive Loss for Collaborative Filtering (CIKM 2023).

    Implements the full PopDCL model from Liu et al., CIKM 2023.  The encoder is
    LightGCN (He et al., SIGIR 2020); the novelty is the loss function that
    simultaneously corrects:
      - **Positive scores** via M+(u,i): reduces the score of positive pairs that
        are likely false-positives due to popularity bias (Sections 3.3, Eq. 3–6).
      - **Negative scores** via M-(u,j): personalizes the debiased contrastive loss
        using a per-user false-negative probability omega+(u) (Section 3.4, Eq. 8–10).

    Both corrections rely solely on item/user popularity (degree in the interaction
    graph), which is pre-computed from the training set and stored as a fixed buffer.

    Args:
        params (dict): Model parameters (see annotated attributes below).
        info (dict): Dataset information dict containing 'n_users' and 'n_items'.
        interactions (Interactions): Training interactions used to build the graph
            adjacency matrix and precompute popularity statistics.
        *args (Any): Variable length argument list (forwarded to LightningModule).
        seed (int): Random seed for reproducibility. Default: 42.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: POS_NEG_LOADER – yields (user, pos_item, neg_item) triplets.
            The neg_item column is not used by the loss; in-batch negatives are derived
            from the pos_item column of each mini-batch (see Section 3.2).
        embedding_size (int): Dimensionality of user/item embedding vectors.
        n_layers (int): Number of LightGCN propagation layers.
        temperature (float): Contrastive temperature parameter tau (Section 3.6).
        reg_weight (float): L2 regularization coefficient lambda (Eq. 16).
        batch_size (int): Training mini-batch size.
        epochs (int): Number of training epochs.
        learning_rate (float): Adam learning rate.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    temperature: float  # tau in the paper (Section 3.6)
    reg_weight: float  # lambda in Eq. 16
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

        # ---- Embedding tables (same structure as LightGCN) ----
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # ---- Graph adjacency matrix (un-normalized, same as LightGCN) ----
        # ASSUMPTION: LightGCN is used as the GNN encoder; the adjacency matrix is
        # built identically to the canonical LightGCN implementation in WarpRec.
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,  # +1 for padding index
        )

        # ---- LGConv propagation network (K layers) ----
        propagation_network_list = []
        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(), "x, edge_index -> x"))
        self.propagation_network = torch_geometric.nn.Sequential(
            "x, edge_index", propagation_network_list
        )

        # ---- Precompute popularity statistics from training interactions ----
        # pop(u) = degree of user u  (Section 3.1)
        # pop(i) = degree of item i  (Section 3.1)
        # Both defined as the number of observed interactions (row / column sums
        # of the binary interaction matrix).
        sparse_matrix = interactions.get_sparse()  # shape: [n_users, n_items]

        # User popularity: row sums → [n_users]
        user_pop_np = np.asarray(sparse_matrix.sum(axis=1)).flatten()  # Eq. 3 denom
        # Item popularity: column sums → [n_items]
        item_pop_np = np.asarray(sparse_matrix.sum(axis=0)).flatten()  # Eq. 3 num

        # Total number of interactions N (Eq. 8 denominator)
        # ASSUMPTION: N is the total training set size (nnz of the sparse matrix).
        total_interactions = float(sparse_matrix.nnz)

        # Register as non-trainable buffers so they move to the correct device
        # automatically when model.to(device) is called.
        self.register_buffer(
            "user_pop",
            torch.tensor(user_pop_np, dtype=torch.float32),
        )  # shape: [n_users]
        self.register_buffer(
            "item_pop",
            torch.tensor(
                np.concatenate([item_pop_np, [0.0]]),  # pad slot → pop 0
                dtype=torch.float32,
            ),
        )  # shape: [n_items + 1]

        # ---- omega+(u) = sum_{i in N_u} pop(i) / N  (Eq. 8) ----
        # Precomputed per user; shape: [n_users].
        # This is the personalized false-negative probability.
        # We clamp to [eps, 1 - eps] to keep omega-(u) > 0.
        user_interacted_item_pop_sum = np.array(
            sparse_matrix.multiply(
                np.asarray(sparse_matrix.sum(axis=0))  # item_pop broadcast
            ).sum(axis=1)
        ).flatten()
        # SIMPLIFICATION: sparse.multiply broadcasts item pop as column vector.
        # Equivalent to dot(R_u, pop_item) for each user u, which equals
        # sum_{i in N_u} pop(i).  Result: [n_users].
        omega_plus_np = user_interacted_item_pop_sum / total_interactions
        omega_plus_np = np.clip(omega_plus_np, 1e-7, 1.0 - 1e-7)
        self.register_buffer(
            "omega_plus",
            torch.tensor(omega_plus_np, dtype=torch.float32),
        )  # shape: [n_users]

        # ---- sum_{i' in N_u} pop(i') per user  (Eq. 3 denominator) ----
        # Used inside the loss to compute P(i ∉ N_u).
        # Already captured as user_interacted_item_pop_sum above.
        user_pop_sum_np = user_interacted_item_pop_sum  # same quantity
        self.register_buffer(
            "user_pop_sum",
            torch.tensor(user_pop_sum_np, dtype=torch.float32),
        )  # shape: [n_users]

        # ---- Weight initialization (Xavier, as stated in Section 4.1.4) ----
        self.apply(self._init_weights)

        # ---- Regularization loss ----
        self.reg_loss = EmbLoss()

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

    def forward(self) -> Tuple[Tensor, Tensor]:
        """LightGCN propagation: layer-wise mean pooling of embeddings.

        Returns:
            Tuple[Tensor, Tensor]: (user_all_embeddings, item_all_embeddings).
        """
        ego_embeddings = self.get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )

        # Move adjacency to the same device as embeddings (lazy migration)
        if self.adj.device() != ego_embeddings.device:
            self.adj = self.adj.to(ego_embeddings.device)

        embeddings_list = [ego_embeddings]
        current_embeddings = ego_embeddings
        for layer_module in self.propagation_network.children():
            current_embeddings = layer_module(current_embeddings, self.adj)
            embeddings_list.append(current_embeddings)

        # Mean pooling across layers 0..K  (LightGCN Eq. 11 / Section 3.2.1)
        lightgcn_all_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)

        # Split into user and item sub-matrices
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings,
            [self.n_users, self.n_items + 1],
        )
        return user_all_embeddings, item_all_embeddings

    def _compute_popdcl_loss(
        self,
        user_emb: Tensor,
        item_emb_batch: Tensor,
        user_idx: Tensor,
        pos_item_idx: Tensor,
    ) -> Tensor:
        """Compute the PopDCL contrastive loss for one mini-batch.

        Args:
            user_emb (Tensor): L2-normalized user embeddings for the batch users. shape: [B, d].
            item_emb_batch (Tensor): L2-normalized positive item embeddings; items
                are also used as in-batch negatives for every other user in the batch. shape: [B, d].
            user_idx (Tensor): Integer user IDs (used to fetch popularity stats). shape: [B]
            pos_item_idx (Tensor):  Integer positive-item IDs. shape: [B]

        Returns:
            Tensor: Scalar loss averaged over the batch.
        """
        # B = batch size
        B = user_emb.size(0)

        # -- Score matrix: f(u_b, i_k) for all (b, k) pairs in batch  (Eq. 1) --
        # shape: [B, B]
        # ASSUMPTION: cosine similarity is used as f(u, i) (Section 3.2, just before Eq. 2).
        # Since both embeddings are L2-normalized, cosine = dot-product.
        scores = torch.mm(user_emb, item_emb_batch.t())  # [B, B]

        # Positive scores are on the diagonal: f(u_b, i_b)
        pos_scores = torch.diag(scores)  # [B]

        # ----------------------------------------------------------------
        # Positive score correction: M+(u, i)   (Eqs. 3, 5, 6)
        # ----------------------------------------------------------------

        # f^-(u, i) = mean of scores to all in-batch negatives j ≠ i  (Eq. 5)
        # For efficiency, subtract the positive score and divide by (B-1).
        # SIMPLIFICATION: The paper says the sum is over j ∈ B\{i} but does not
        # explicitly handle B=1 edge cases; we guard with clamp.
        row_sum = scores.sum(dim=1)  # [B] – sum over all j including i
        f_neg_u = (row_sum - pos_scores) / max(B - 1, 1)  # [B] – Eq. 5

        # P(i ∉ N_u) = pop(i) / sum_{i' ∈ N_u} pop(i')   (Eq. 3)
        pop_pos = self.item_pop[pos_item_idx]  # type: ignore[index]
        user_pop_sum_b = self.user_pop_sum[user_idx]  # type: ignore[index]
        # Guard against zero denominator (new users with no interactions)
        # ASSUMPTION: user_pop_sum is always > 0 for users in training set.
        p_false_pos = pop_pos / user_pop_sum_b.clamp(min=1.0)  # [B] – Eq. 3

        # M+(u, i) = sigma(P(i ∉ N_u) * f^-(u, i))   (Eq. 6)
        m_plus = torch.sigmoid(p_false_pos * f_neg_u)  # [B] – Eq. 6

        # ----------------------------------------------------------------
        # Negative score correction: M-(u, j)   (Eqs. 8, 10)
        # ----------------------------------------------------------------

        # omega+(u) per user in batch  (Eq. 8, precomputed)
        omega_p = self.omega_plus[user_idx]  # type: ignore[index]
        omega_m = 1.0 - omega_p  # [B] – omega-(u)

        # M-(u, j) = omega+(u) / omega-(u) * exp(1/tau * [f(u,i) - f(u,j)])  (Eq. 10)
        # shape: [B, B]  – one row per user, one column per in-batch negative j
        #
        # The Maclaurin expansion in the paper yields this closed-form for M-(u,j).
        # pos_scores[:, None]  → [B, 1] broadcasts over the j dimension.
        ratio = (omega_p / omega_m).unsqueeze(1)  # [B, 1]
        delta_f = pos_scores.unsqueeze(1) - scores  # [B, B] – f(u,i) - f(u,j)
        m_minus = ratio * torch.exp(delta_f / self.temperature)  # [B, B] – Eq. 10

        # ----------------------------------------------------------------
        # Full PopDCL loss  (Eq. 1, rewritten in log form for stability)
        # ----------------------------------------------------------------
        # L(u, i) = log(1 + sum_{j ≠ i} exp(1/tau * [f(u,j) - M-(u,j)
        #                                              - (f(u,i) - M+(u,i))]))
        #         = log(1 + sum_{j ≠ i} exp(1/tau * [Δf - ΔM]))   (Eq. 14)
        #
        # where:
        #   corrected_pos_score[b]   = (pos_scores[b] - m_plus[b]) / tau
        #   corrected_neg_score[b,k] = (scores[b,k]   - m_minus[b,k]) / tau  for k ≠ b

        # Numerator in Eq. 1 (in log-space):
        #   log_num[b] = (f(u_b, i_b) - M+(u_b, i_b)) / tau
        log_num = (pos_scores - m_plus) / self.temperature  # [B]

        # Build corrected negative logits: (f(u,j) - M-(u,j)) / tau
        corrected_neg = (scores - m_minus) / self.temperature  # [B, B]

        # Mask out the positive pair (diagonal) so it doesn't appear in the sum
        # over negatives.
        diag_mask = torch.eye(B, dtype=torch.bool, device=scores.device)
        corrected_neg = corrected_neg.masked_fill(diag_mask, float("-inf"))

        # Denominator in Eq. 1 (numerator + sum of negatives), computed via
        # logsumexp for numerical stability:
        #   log_denom[b] = logsumexp(
        #       [log_num[b], corrected_neg[b,0], ..., corrected_neg[b,B-1]]
        #   )
        all_logits = torch.cat([log_num.unsqueeze(1), corrected_neg], dim=1)  # [B, B+1]
        log_denom = torch.logsumexp(all_logits, dim=1)  # [B]

        # Per-sample loss: - log( exp(log_num) / exp(log_denom) )
        #                = log_denom - log_num
        loss_per_sample = log_denom - log_num  # [B] – Eq. 1

        return loss_per_sample.mean()

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """One training iteration.

        The standard WarpRec contrastive dataloader provides
        (user, pos_item, neg_item) triplets.  neg_item is ignored here because
        PopDCL constructs negatives in-batch from the pos_item column.

        Args:
            batch (Any): Triplet (user [B], pos_item [B], neg_item [B]).
            batch_idx (int): Current batch index.

        Returns:
            Tensor: Scalar training loss.
        """
        user, pos_item, _ = batch  # neg_item unused — in-batch strategy

        # Get full propagated embeddings from LightGCN encoder
        user_all_embeddings, item_all_embeddings = self.forward()

        # Gather per-batch embeddings
        u_emb = user_all_embeddings[user]  # [B, d]
        i_emb = item_all_embeddings[pos_item]  # [B, d]

        # L2-normalize: required by Eq. 5 and stated explicitly in Section 3.3
        # ("we use normalization to stabilize contrastive learning, meaning that
        # the user embedding e_u and item embedding e_i are both l2-normalized")
        u_emb = F.normalize(u_emb, p=2, dim=-1)  # Sec. 3.3
        i_emb = F.normalize(i_emb, p=2, dim=-1)  # Sec. 3.3

        # --- PopDCL contrastive loss (Eqs. 1–10, 14) ---
        pop_loss = self._compute_popdcl_loss(
            user_emb=u_emb,
            item_emb_batch=i_emb,
            user_idx=user,
            pos_item_idx=pos_item,
        )

        # --- L2 regularization on initial (un-propagated) embeddings (Eq. 16) ---
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
        )  # Eq. 16

        loss = pop_loss + reg_loss
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute recommendation scores for the given users.

        At inference time, scores are plain cosine similarities between the
        propagated user and item embeddings (no correction applied).
        ASSUMPTION: Corrections are applied only during training to debias the
        loss; the final embeddings are used directly for ranking (consistent with
        how BC_loss, DCL, HCL are evaluated — all use plain inner-product/cosine
        at test time).

        Args:
            user_indices (Tensor): Batch of user IDs.
            *args (Any): Unused positional arguments.
            item_indices (Optional[Tensor]): If None, scores against all items are
                returned.  Otherwise, scores for the sampled item sub-set.
            **kwargs (Any): Unused keyword arguments.

        Returns:
            Tensor: Score matrix [batch_size, n_items] or [batch_size, n_samples].
        """
        user_all_embeddings, item_all_embeddings = self.propagate_embeddings()

        user_embeddings = user_all_embeddings[user_indices]  # [B, d]

        if item_indices is None:
            # Full ranking – score against all n_items (drop padding slot)
            item_embeddings = item_all_embeddings[:-1, :]  # [n_items, d]
            return torch.einsum("be,ie->bi", user_embeddings, item_embeddings)

        # Sampled ranking
        item_embeddings = item_all_embeddings[item_indices]  # [B, S, d]
        return torch.einsum("be,bse->bs", user_embeddings, item_embeddings)
