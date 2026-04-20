# pylint: disable = R0801, E1102
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn, Tensor
from torch_geometric.nn import LGConv
from torch.utils.data import DataLoader

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="PAAC")
class PAAC(GraphRecommenderUtils, IterativeRecommender):
    """Implementation of PAAC from
    "Popularity-Aware Alignment and Contrast for Mitigating Popularity Bias in Recommendation" (KDD 2024).

    Popularity-Aware Alignment and Contrast (PAAC) for Mitigating Popularity Bias.

    PAAC wraps a LightGCN backbone with two debiasing modules:

    1. **Supervised Alignment Module** (Section 3.1): For each user in the batch,
       popular items (higher interaction frequency) and unpopular items are identified
       among the user's interactions. A Frobenius-norm alignment loss pulls the
       representations of co-interacted popular and unpopular items together,
       injecting additional supervision into the sparse unpopular embeddings.

    2. **Re-weighting Contrast Module** (Section 3.2): Items in each mini-batch are
       split into a popular group (top x%) and an unpopular group (bottom (1-x)%).
       Two InfoNCE losses — one per group as positive anchor — are computed with
       *asymmetric* negative weighting: the ``beta`` hyperparameter down-weights the
       cross-group negatives so that popular and unpopular items are not pushed
       excessively apart. The two losses are combined via ``gamma`` (Eq. 7).
       Noise perturbation generates the second augmented view (Section 2.2).

    The total loss follows Eq. 11:
        L = L_rec + λ₁ * L_sa + λ₂ * L_cl + λ₃ * ||Θ||²

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

        # ------------------------------------------------------------------ #
        # LightGCN propagation layers — Eq. LightGCN (Section 2.1)           #
        # ------------------------------------------------------------------ #
        propagation_network_list = []
        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(), "x, edge_index -> x"))
        self.propagation_network = torch_geometric.nn.Sequential(
            "x, edge_index", propagation_network_list
        )

        # Normalized adjacency matrix for LightGCN propagation
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,
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

        # Also compute user popularity for the user-side contrastive loss
        user_pop = torch.tensor(sparse_mat.sum(axis=1).A1, dtype=torch.float32)
        self.register_buffer("user_popularity", user_pop)

        # ------------------------------------------------------------------ #
        # Historical interaction data for supervised alignment module (Section 3.1) #
        # ------------------------------------------------------------------ #
        # ASSUMPTION: we use the full user-item interaction history from the training set
        # for the supervised alignment loss, as described in the paper.
        # This is stored as buffers to avoid redundant computation and ensure it moves with the model.
        history_matrix, history_lens, _ = interactions.get_history()
        self.register_buffer("history_matrix", history_matrix)
        self.register_buffer("history_lens", history_lens)

        # ------------------------------------------------------------------- #
        # Precompute popular/unpopular item splits for each user based on their interaction history.
        # This allows efficient retrieval during the supervised alignment loss computation.
        # -------------------------------------------------------------------- #
        pop_matrix = torch.zeros_like(history_matrix)
        unpop_matrix = torch.zeros_like(history_matrix)
        pop_lens = torch.zeros_like(history_lens)
        unpop_lens = torch.zeros_like(history_lens)

        for u in range(self.n_users):
            length = history_lens[u]
            if length < 2:
                continue

            items = history_matrix[u, :length]
            pops = item_pop[items]  # Get popularity of the user's interacted items

            median_pop = pops.median()
            pop_mask = pops > median_pop
            unpop_mask = pops <= median_pop

            p_items = items[pop_mask]
            u_items = items[unpop_mask]

            pop_lens[u] = len(p_items)
            unpop_lens[u] = len(u_items)

            if len(p_items) > 0:
                pop_matrix[u, : len(p_items)] = p_items
            if len(u_items) > 0:
                unpop_matrix[u, : len(u_items)] = u_items

        # Register the precomputed popular/unpopular splits as buffers
        self.register_buffer("pop_history_matrix", pop_matrix)
        self.register_buffer("pop_history_lens", pop_lens)
        self.register_buffer("unpop_history_matrix", unpop_matrix)
        self.register_buffer("unpop_history_lens", unpop_lens)

        # ------------------------------------------------------------------ #
        # Loss functions                                                       #
        # ------------------------------------------------------------------ #
        self.bpr_loss = BPRLoss()  # Eq. 1
        self.reg_loss = EmbLoss()  # λ₃ * ||Θ||²  (Eq. 11)

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

    def forward(self, perturbed: bool = False) -> Tuple[Tensor, Tensor]:
        """LightGCN graph propagation with optional noise perturbation.

        Noise perturbation is used to generate the second augmented view for
        the re-weighting contrastive loss (Section 2.2, noise-based augmentation).

        Args:
            perturbed (bool): If True adds uniform noise scaled by ``eps``
                (ASSUMPTION: same noise scheme as SimGCL/XSimGCL — normalized
                uniform noise added to each layer's output).

        Returns:
            Tuple[Tensor, Tensor]: Propagated user embeddings [n_users, D]
                and item embeddings [n_items+1, D].
        """
        ego_embeddings = self.get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )

        # Move adjacency matrix to current device if needed
        if self.adj.device() != ego_embeddings.device:
            self.adj = self.adj.to(ego_embeddings.device)

        embeddings_list = [ego_embeddings]
        current_embeddings = ego_embeddings

        for layer_module in self.propagation_network.children():
            # LightGCN graph convolution: E^(l) = A * E^(l-1)
            current_embeddings = layer_module(current_embeddings, self.adj)
            if perturbed:
                # Noise perturbation for contrastive view (Section 2.2)
                noise = F.normalize(torch.rand_like(current_embeddings), p=2, dim=1)
                current_embeddings = current_embeddings + self.eps * noise
            embeddings_list.append(current_embeddings)

        # Mean-pool across layers (LightGCN aggregation)
        all_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)

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
        """Compute the popularity-aware supervised alignment loss L_sa (Eq. 5).

        For each user u in the batch, we identify the set of items I_u that
        appear as positive samples in this batch. We then split I_u into a
        popular group I_u^pop and an unpopular group I_u^unpop by ranking
        items by their global popularity p(i) (Eq. 4). We minimize the
        squared Frobenius distance between all (popular, unpopular) pairs
        of item representations:

            L_sa = Σ_u  1/|I_u|  Σ_{i ∈ I_u^pop, i' ∈ I_u^unpop}  ||h_i - h_{i'}||²

        SIMPLIFICATION: We approximate I_u with the positive items that
        co-occur in the same batch for the same user.  In the original code
        the full I_u from the training set is used; this batch-level
        approximation avoids an expensive global lookup per step while still
        injecting the supervision signal as described in the paper.

        Args:
            users (Tensor): User indices in the batch, shape [B].
            pos_items (Tensor): Positive item indices in the batch, shape [B].
            item_embeddings (Tensor): Propagated item embeddings [n_items+1, D].

        Returns:
            Tensor: Scalar supervised alignment loss.
        """
        device = item_embeddings.device
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        # Retrieve only the unique users in the batch to avoid redundant computation
        unique_users = users.unique()

        for u_idx in unique_users:
            # Get the length of the user's interaction history in the training set
            N = self.pop_history_lens[u_idx]  # type: ignore[index]
            M = self.unpop_history_lens[u_idx]  # type: ignore[index]

            if N == 0 or M == 0:
                continue

            # Retrieve the popular and unpopular item indices for this user from the precomputed buffers
            pop_items = self.pop_history_matrix[u_idx, :N]  # type: ignore[index]
            unpop_items = self.unpop_history_matrix[u_idx, :M]  # type: ignore[index]

            h_pop = item_embeddings[pop_items]  # [n_pop, D]
            h_unpop = item_embeddings[unpop_items]  # [n_unpop, D]

            # OPTIMIZATION
            sum_sq_pop = (h_pop**2).sum()
            sum_sq_unpop = (h_unpop**2).sum()

            sum_pop = h_pop.sum(dim=0)  # [D]
            sum_unpop = h_unpop.sum(dim=0)  # [D]

            dot_product = torch.dot(sum_pop, sum_unpop)

            pair_loss_sum = M * sum_sq_pop + N * sum_sq_unpop - 2 * dot_product
            pair_loss_sum = torch.clamp(pair_loss_sum, min=0.0)  # Numerical stability

            n_I_u = (N + M).float()
            total_loss = total_loss + (pair_loss_sum / n_I_u)
            count += 1

        if count == 0:
            return total_loss

        return total_loss / count

    def _reweighting_contrast_loss(
        self,
        item_emb_view1: Tensor,
        item_emb_view2: Tensor,
        user_emb_view1: Tensor,
        user_emb_view2: Tensor,
        batch_items: Tensor,
        batch_users: Tensor,
    ) -> Tensor:
        """Compute the popularity-centric re-weighting contrastive loss L_cl (Eq. 10).

        Step 1 — Dynamic batch-level popularity split (Eq. 6):
            Items in the batch are ranked by global popularity; the top
            ``pop_ratio`` fraction form I^pop and the rest form I^unpop.

        Step 2 — Per-group InfoNCE with asymmetric β weighting (Eq. 8/9):
            For popular items as positive anchors:
                L^pop  = Σ_{i ∈ I^pop} log exp(h'_i h''_i / τ)
                              / [ Σ_{j ∈ I^pop} exp(h'_i h''_j / τ)
                                 + β Σ_{j ∈ I^unpop} exp(h'_i h''_j / τ) ]

            For unpopular items as positive anchors:
                L^unpop = Σ_{i ∈ I^unpop} log exp(h'_i h''_i / τ)
                              / [ Σ_{j ∈ I^unpop} exp(h'_i h''_j / τ)
                                 + β Σ_{j ∈ I^pop} exp(h'_i h''_j / τ) ]

        Step 3 — Combine item and user objectives (Eq. 10):
            L^item_cl = γ * L^pop + (1 - γ) * L^unpop
            L_cl = 0.5 * (L^item_cl + L^user_cl)

        ASSUMPTION: The user-side contrastive loss L^user_cl follows the same
        structure as L^item_cl using the corresponding user embeddings from the
        two augmented views.  The paper states "the final re-weighting contrastive
        objective is the weighted sum of the user objective and the item objective"
        (Eq. 10) but does not detail the user-side formula — we apply the same
        γ/β re-weighting using batch users split by their own interaction count.

        Args:
            item_emb_view1 (Tensor): Propagated item embeddings, view 1 [n_items+1, D].
            item_emb_view2 (Tensor): Propagated item embeddings, view 2 (perturbed) [n_items+1, D].
            user_emb_view1 (Tensor): Propagated user embeddings, view 1 [n_users, D].
            user_emb_view2 (Tensor): Propagated user embeddings, view 2 (perturbed) [n_users, D].
            batch_items (Tensor): Unique item indices in the batch [m].
            batch_users (Tensor): Unique user indices in the batch [p].

        Returns:
            Tensor: Scalar re-weighting contrastive loss.
        """
        # ---- Item-side contrastive loss ----------------------------------- #
        # Gather batch-item embeddings from both views
        h_prime = item_emb_view1[batch_items]  # [m, D] — Eq. 8/9: h'
        h_dprime = item_emb_view2[batch_items]  # [m, D] — Eq. 8/9: h''

        # L2 normalize for cosine similarity (standard InfoNCE practice)
        h_prime_n = F.normalize(h_prime, p=2, dim=1)
        h_dprime_n = F.normalize(h_dprime, p=2, dim=1)

        # Eq. 6: dynamic batch-level popularity split
        pops = self.item_popularity[batch_items]  # type: ignore[index]
        k_pop = max(1, int(self.pop_ratio * batch_items.numel()))
        _, sorted_idx = pops.sort(descending=True)
        pop_mask = torch.zeros(
            batch_items.numel(), dtype=torch.bool, device=pops.device
        )
        pop_mask[sorted_idx[:k_pop]] = True
        unpop_mask = ~pop_mask

        item_cl_loss = self._group_infonce(
            h_prime_n, h_dprime_n, pop_mask, unpop_mask
        )  # Eq. 7

        # ---- User-side contrastive loss ----------------------------------- #
        # ASSUMPTION: popularity split for users uses their interaction count
        # (analogous to item popularity), derived from the item_popularity buffer
        # reinterpreted for users — we use the user's positive-item batch count
        # as a proxy, which is always equal (1 per user in POS_NEG_LOADER).
        # Therefore we apply a uniform 50/50 split among batch users.
        z_prime = user_emb_view1[batch_users]  # [p, D]
        z_dprime = user_emb_view2[batch_users]  # [p, D]
        z_prime_n = F.normalize(z_prime, p=2, dim=1)
        z_dprime_n = F.normalize(z_dprime, p=2, dim=1)

        # Dynamic batch-level popularity split for users based on their interaction count
        u_pops = self.user_popularity[batch_users]  # type: ignore[index]
        k_pop_u = max(1, int(self.pop_ratio * batch_users.numel()))
        _, sorted_idx_u = u_pops.sort(descending=True)

        u_pop_mask = torch.zeros(
            batch_users.numel(), dtype=torch.bool, device=z_prime.device
        )
        u_pop_mask[sorted_idx_u[:k_pop_u]] = True
        u_unpop_mask = ~u_pop_mask

        user_cl_loss = self._group_infonce(
            z_prime_n, z_dprime_n, u_pop_mask, u_unpop_mask
        )

        # Eq. 10: L_cl = 0.5 * (L^item_cl + L^user_cl)
        return 0.5 * (item_cl_loss + user_cl_loss)

    def _group_infonce(
        self,
        view1: Tensor,
        view2: Tensor,
        pop_mask: Tensor,
        unpop_mask: Tensor,
    ) -> Tensor:
        """Compute γ-weighted InfoNCE for the two popularity groups.

        Implements Eq. 7: L^item_cl = γ * L^pop + (1 - γ) * L^unpop
        using the β-re-weighted denominators from Eq. 8 and Eq. 9.

        Args:
            view1 (Tensor): L2-normalized embeddings, view 1 [N, D].
            view2 (Tensor): L2-normalized embeddings, view 2 [N, D].
            pop_mask (Tensor): Boolean mask for popular items/users [N].
            unpop_mask (Tensor): Boolean mask for unpopular items/users [N].

        Returns:
            Tensor: Scalar γ-weighted group InfoNCE loss.
        """
        # Precompute log(beta) for the cross-group weighting
        log_beta = torch.log(
            torch.tensor(self.beta, device=view1.device).clamp(min=1e-8)
        )

        # ---- L^pop (Eq. 8) --- popular items as positive samples ---------- #
        if pop_mask.sum() > 0:
            h_pop_v1 = view1[pop_mask]  # [n_pop, D]
            h_pop_v2 = view2[pop_mask]  # [n_pop, D]
            h_unpop_v2 = view2[unpop_mask]  # [n_unpop, D]

            # Positive scores: exp(h'_i h''_i / τ) for i ∈ I^pop
            pos_scores_pop = (h_pop_v1 * h_pop_v2).sum(
                dim=1
            ) / self.temperature  # [n_pop]

            # Intra-group denominator: Σ_{j ∈ I^pop} exp(h'_i h''_j / τ)
            intra_pop = (
                torch.matmul(h_pop_v1, h_pop_v2.T) / self.temperature
            )  # [n_pop, n_pop]
            log_intra_pop = torch.logsumexp(intra_pop, dim=1)

            # Cross-group denominator: β * Σ_{j ∈ I^unpop} exp(h'_i h''_j / τ)
            if h_unpop_v2.numel() > 0:
                cross_pop = (
                    torch.matmul(h_pop_v1, h_unpop_v2.T) / self.temperature
                )  # [n_pop, n_unpop]
                log_cross_pop = log_beta + torch.logsumexp(cross_pop, dim=1)

                # Correctly compute log(X + beta * Y) using logaddexp
                denom_pop = torch.logaddexp(log_intra_pop, log_cross_pop)
            else:
                denom_pop = log_intra_pop

            l_pop = -(pos_scores_pop - denom_pop).mean()  # Eq. 8
        else:
            l_pop = torch.tensor(0.0, device=view1.device)

        # ---- L^unpop (Eq. 9) — unpopular items as positive samples ------- #
        if unpop_mask.sum() > 0:
            h_unpop_v1 = view1[unpop_mask]  # [n_unpop, D]
            h_unpop_v2 = view2[unpop_mask]  # [n_unpop, D]
            h_pop_v2_for_unpop = view2[pop_mask]  # [n_pop, D]

            # Positive scores for unpopular
            pos_scores_unpop = (h_unpop_v1 * h_unpop_v2).sum(
                dim=1
            ) / self.temperature  # [n_unpop]

            # Intra-group denominator: Σ_{j ∈ I^unpop}
            intra_unpop = (
                torch.matmul(h_unpop_v1, h_unpop_v2.T) / self.temperature
            )  # [n_unpop, n_unpop]
            log_intra_unpop = torch.logsumexp(intra_unpop, dim=1)

            # Cross-group: β * Σ_{j ∈ I^pop}
            if h_pop_v2_for_unpop.numel() > 0:
                cross_unpop = (
                    torch.matmul(h_unpop_v1, h_pop_v2_for_unpop.T) / self.temperature
                )  # [n_unpop, n_pop]
                log_cross_unpop = log_beta + torch.logsumexp(cross_unpop, dim=1)

                # Correctly compute log(X + beta * Y) using logaddexp
                denom_unpop = torch.logaddexp(log_intra_unpop, log_cross_unpop)
            else:
                denom_unpop = log_intra_unpop

            l_unpop = -(pos_scores_unpop - denom_unpop).mean()  # Eq. 9
        else:
            l_unpop = torch.tensor(0.0, device=view1.device)

        # Eq. 7: L^item_cl = γ * L^pop + (1 - γ) * L^unpop
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

        # ---- Step 4 (Alg. 1): compute user and item representations ------- #
        # View 1: clean propagation (for recommendation loss + alignment)
        user_emb, item_emb = self.forward(perturbed=False)

        # View 2: noise-perturbed propagation (for contrastive loss)
        user_emb_p, item_emb_p = self.forward(perturbed=True)

        # ---- Step 10 (Alg. 1): recommendation loss L_rec (Eq. 1) --------- #
        u_emb = user_emb[user]
        pos_emb = item_emb[pos_item]
        neg_emb = item_emb[neg_item]

        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
        rec_loss = self.bpr_loss(pos_scores, neg_scores)

        # ---- Step 7 (Alg. 1): supervised alignment loss L_sa (Eq. 5) ----- #
        # Alignment is performed on the propagated representations h_i = f(i)
        sa_loss = self._supervised_alignment_loss(user, pos_item, item_emb)

        # ---- Step 9 (Alg. 1): re-weighting contrastive loss L_cl (Eq. 10) #
        # ASSUMPTION: we compute the contrastive loss over the unique items and
        # users that appear in the batch, which captures the batch-level interaction
        # structure. The original paper states "the contrastive loss is computed over all
        # items/users in the batch" but does not detail wether negatives are included.
        batch_items = torch.cat([pos_item, neg_item]).unique()
        batch_users = user.unique()
        cl_loss = self._reweighting_contrast_loss(
            item_emb, item_emb_p, user_emb, user_emb_p, batch_items, batch_users
        )

        # ---- L2 regularization — λ₃ * ||Θ||² (Eq. 11) ------------------- #
        # ASSUMPTION: regularize over initial (ego) embeddings only, consistent
        # with "Θ is the set of model parameters in L_rec" (Section 3.3).
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        # ---- Total loss (Eq. 11) ----------------------------------------- #
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
