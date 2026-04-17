# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional

import torch
import torch_geometric
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import LGConv

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class Projector(nn.Module):
    """Two-layer MLP projector used in the BCL objective.

    Architecture from Figure 3 (Appendix D.4):
        MLP → BatchNorm → ReLU → MLP → BatchNorm

    This is the h(·) function in Eq. 9.

    Args:
        dim (int): Input and output dimensionality (equal to embedding_size).

    Notes:
        - ASSUMPTION: hidden size equals the embedding size (dim). The paper
          does not specify a hidden dimension beyond "MLP", and the original
          SimSiam projector uses hidden == output == input for recommendation.
        - SIMPLIFICATION: A single intermediate projection with hidden == dim
          is used; the paper only shows a schematic (two MLP blocks + norms).
    """

    def __init__(self, dim: int):
        super().__init__()
        # Figure 3 — MLP → BN → ReLU → MLP → BN
        self.net = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project embeddings through the two-layer MLP.

        Args:
            x (Tensor): Input embeddings [B, dim].

        Returns:
            Tensor: Projected embeddings [B, dim].
        """
        return self.net(x)


@model_registry.register(name="RecDCL")
class RecDCL(GraphRecommenderUtils, IterativeRecommender):
    """Implementation of RecDCL model
    from "RecDCL: Dual Contrastive Learning for Recommendation" (WWW 2024).

    Implements the full RecDCL framework (Zhang et al., WWW 2024) which combines:
      - **FCL objective** (feature-wise CL):
          - UIBT: Barlow-Twins-style cross-correlation loss between user and item
            embeddings to eliminate inter-user/item redundancy (Eq. 5).
          - UUII: Polynomial-kernel uniformity loss applied *within* the user
            and item embedding matrices along the feature dimension (Eq. 6).
      - **BCL objective** (batch-wise CL):
          - Historical-embedding output augmentation inspired by SimSiam, using
            online and target networks with shared graph encoder (Eqs. 8–9).

    The total training objective is (Eq. 10):
        L = L_UIBT + alpha * L_UUII + beta * L_BCL

    The encoder is a 2-layer LightGCN (He et al., SIGIR 2020).  Embeddings are
    L2-normalized before all loss computations (Algorithm 1).

    Args:
        params (dict): Model parameters (see annotated attributes below).
        info (dict): Dataset information dict containing 'n_users' and 'n_items'.
        interactions (Interactions): Training interactions used to build the
            graph adjacency matrix.
        *args (Any): Variable length argument list (forwarded to LightningModule).
        seed (int): Random seed for reproducibility. Default: 42.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: POS_LOADER – yields (user, pos_item) pairs.
            RecDCL does not require negative sampling (Section 4.2 / Table 1).
        embedding_size (int): Dimensionality of user/item embedding vectors (F).
            Best results at large F (2048); see Section D.7 / Figure 4.
        n_layers (int): Number of LightGCN propagation layers. Default: 2.
        gamma (float): Redundancy-reduction weight in UIBT (Eq. 5). Default: 0.01.
        alpha (float): Coefficient for UUII loss in total objective (Eq. 10).
            Default: 0.2 (best for Beauty/Game; see Table 13 and Figure 5).
        beta (float): Coefficient for BCL loss in total objective (Eq. 10).
            Default: 5 (best for Beauty/Yelp; see Table 13 and Figure 6).
        tau_momentum (float): Momentum ratio for historical embedding blending
            in Eq. 8. Default: 0.1 (best for Beauty; Table 13 / Figure 8).
        poly_a (float): Polynomial kernel coefficient a for UUII (Eq. 6). Default: 1.
        poly_c (float): Polynomial kernel offset c for UUII (Eq. 6). Default: 1e-7.
        poly_e (int): Polynomial kernel exponent e for UUII (Eq. 6). Default: 4.
        batch_size (int): Training mini-batch size. Default 256 for Beauty, 1024
            for Food/Game/Yelp (Appendix D.3).
        epochs (int): Number of training epochs.
        learning_rate (float): Adam learning rate. Default: 0.001 (Appendix D.3).
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_DATALOADER

    # Model hyperparameters
    embedding_size: int
    n_layers: int
    gamma: float
    alpha: float
    beta: float
    tau_momentum: float
    poly_a: float
    poly_c: float
    poly_e: int
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

        # ---- Embedding tables (same structure as LightGCN) ---- Algorithm 1
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # ---- Graph adjacency matrix ----
        # ASSUMPTION: The LightGCN adjacency is built without explicit symmetric
        # normalization here (same as the canonical WarpRec LightGCN); LGConv
        # internally applies the degree normalization during message passing.
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,  # +1 for padding idx
        )

        # ---- LGConv propagation network (L layers) ----
        # Section 4.2: "we adopt [LightGCN] as the graph encoder f_theta"
        propagation_network_list = []
        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(), "x, edge_index -> x"))
        self.propagation_network = torch_geometric.nn.Sequential(
            "x, edge_index", propagation_network_list
        )

        # ---- BCL projector h(·) — Figure 3 / Eq. 9 ----
        self.projector = Projector(self.embedding_size)

        # ---- Historical embedding buffers for BCL (Eq. 8) ----
        # Initialized to zeros; updated at the end of each forward pass.
        # ASSUMPTION: "historical embeddings from prior training iterations"
        # (Section 4.2) means the embeddings from the immediately preceding
        # batch/step, stored as a detached copy.  This matches the description
        # in [3, 9, 54] cited in the paper and the GNNAutoScale approach.
        self.register_buffer(
            "hist_user_emb",
            torch.zeros(self.n_users, self.embedding_size),
        )
        self.register_buffer(
            "hist_item_emb",
            torch.zeros(self.n_items + 1, self.embedding_size),
        )
        # Track whether historical buffers have been populated yet
        self._hist_initialized: bool = False

        # ---- Xavier weight initialization ---- (Appendix D.3 / [11])
        self.apply(self._init_weights)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_positive_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def _uibt_loss(
        self,
        e_u: Tensor,
        e_i: Tensor,
        gamma: float,
    ) -> Tensor:
        """Feature-wise cross-correlation loss between users and items (UIBT).

        Extends Barlow Twins [47] to user-item alignment for recommendations.
        The cross-correlation matrix C is built between the user embedding matrix
        E_U and item embedding matrix E_I along the feature dimension.

        The loss has two terms scaled by 1/F (Eq. 5):
        - Invariance term : (1 - C_mm)^2  — drives diagonal to 1.
        - Redundancy term : gamma * C_mn^2 for m != n — drives off-diagonal to 0.

        Args:
            e_u (Tensor): L2-normalized user embeddings  [B, F].
            e_i (Tensor): L2-normalized item embeddings  [B, F].
            gamma (float): Weight on the redundancy-reduction term (Eq. 5).

        Returns:
            Tensor: Scalar UIBT loss.
        """
        # Eq. 5 — Section 4.1 ("Eliminate redundancy between users and items")
        B, F = e_u.shape

        # Cross-correlation matrix C [F, F] — Eq. 3 / Eq. 5
        # C_mn = (E_U[:, m])^T E_I[:, n] / B
        # ASSUMPTION: batch-normalization of the projector output (see Figure 3)
        # means embeddings reaching here are approximately zero-mean.
        # The paper divides by B (batch size) rather than by ||Z^:m|| * ||Z_hat^:n||
        # as in the original Barlow Twins; Algorithm 2 shows: C = mm(e_u.T, e_i).div(B)
        C = torch.mm(e_u.t(), e_i) / B  # [F, F] — Algorithm 2

        # Invariance term: (1 - C_mm)^2 summed, divided by F — Eq. 5
        on_diag = torch.diagonal(C).add_(-1).pow_(2).sum().div(F)

        # Redundancy reduction term: gamma * sum_{m != n} C_mn^2 / F — Eq. 5
        # off_diagonal mask: all elements minus diagonal
        off_diag = C.clone()
        diag_idx = torch.arange(F, device=C.device)
        off_diag[diag_idx, diag_idx] = 0.0
        off_diag_loss = gamma * off_diag.pow(2).sum().div(F)

        return on_diag + off_diag_loss  # Eq. 5

    def _uuii_loss(
        self,
        e_u: Tensor,
        e_i: Tensor,
        poly_a: float,
        poly_c: float,
        poly_e: int,
    ) -> Tensor:
        """Feature-wise uniformity loss within users and within items (UUII).

        Uses a polynomial kernel to drive feature-column representations on the
        user and item hyperspheres toward uniformity. Computed separately for users
        and items, then averaged (Eq. 6 / Section 4.1 "Eliminate redundancy within
        users and items").

        Kernel:  k(z_a, z_b) = (a * z_a^T z_b + c)^e

        The loss is the log of the mean pairwise kernel value over all pairs of
        feature columns within the user (resp. item) embedding matrix.

        Args:
            e_u (Tensor): L2-normalized user embeddings  [B, F].
            e_i (Tensor): L2-normalized item embeddings  [B, F].
            poly_a (float): Polynomial kernel coefficient a. Default 1.
            poly_c (float): Polynomial kernel offset c. Default 1e-7.
            poly_e (int): Polynomial kernel exponent e. Default 4.

        Returns:
            Tensor: Scalar UUII loss.
        """
        # Eq. 6 — Section 4.1
        # Note: the summation/mean is over FEATURE columns (m != n), not batch samples.
        # Algorithm 2: L_uni = mm(e_i.T, e_i).add_(c).pow_(e).mean().log()
        # The 1/2 factor comes from averaging user and item terms.

        def _uni(e: Tensor) -> Tensor:
            # Gram matrix of feature columns: [F, F]
            gram = torch.mm(e.t(), e)  # E^T * E in Algorithm 2
            # Apply polynomial kernel and compute log-mean — Algorithm 2
            return (poly_a * gram + poly_c).pow(poly_e).mean().log()

        return 0.5 * _uni(e_u) + 0.5 * _uni(e_i)  # Eq. 6

    def _bcl_loss(
        self,
        e_u: Tensor,
        e_i: Tensor,
        e_u_hist: Tensor,
        e_i_hist: Tensor,
        tau_momentum: float,
        projector: nn.Module,
    ) -> Tensor:
        """Batch-wise contrastive loss with historical-embedding augmentation (BCL).

        Follows the SimSiam-style asymmetric design (Zhang et al. [48]):
        1. Blend historical and current embeddings to form perturbed views (Eq. 8).
        2. Apply stop-gradient to the perturbed (target-network) side.
        3. Compute cosine distance between the projected current embedding and the
            stop-gradient perturbed embedding (Eq. 9).
        4. Average user→item and item→user directions (Algorithm 2 / Eq. 9).

        Args:
            e_u (Tensor): Current user embeddings  [B, F].
            e_i (Tensor): Current item embeddings  [B, F].
            e_u_hist (Tensor): Historical user embeddings from the target network [B, F].
            e_i_hist (Tensor): Historical item embeddings from the target network [B, F].
            tau_momentum (float): Momentum coefficient tau controlling historical
                embedding weight in Eq. 8.
            projector (nn.Module): The shared MLP projector h(·) (Figure 3).

        Returns:
            Tensor: Scalar BCL loss.
        """
        # Eq. 8 — perturbed augmented representations using historical embeddings
        # e_hat_u = tau * e_u_hist + (1 - tau) * e_u
        e_hat_u = tau_momentum * e_u_hist + (1.0 - tau_momentum) * e_u  # Eq. 8
        e_hat_i = tau_momentum * e_i_hist + (1.0 - tau_momentum) * e_i  # Eq. 8

        # Eq. 9 — asymmetric cosine-distance with stop-gradient on the target side
        # L_BCL = 1/2 * S(h(E_U), sg(E_hat_I)) + 1/2 * S(sg(E_hat_U), h(E_I))
        # S(a, b) = negative cosine similarity (cosine *distance*)
        # ASSUMPTION: cosine distance S = 1 - cos_sim is the intended metric;
        # the paper writes "S(·, ·) denotes the cosine distance" (Section 4.2).
        def _cos_distance(a: Tensor, b: Tensor) -> Tensor:
            return 1.0 - F.cosine_similarity(a, b, dim=-1).mean()

        # h(E_U): projected current user embeddings — applied to current side only
        p_u = projector(e_u)  # online projection
        p_i = projector(e_i)  # online projection

        # Stop gradient on the augmented (target) side — Eq. 9 / Figure 2 (d)
        # Algorithm 2: L_aug = S(h(e_u), sg(e_hat_i))
        loss_u = _cos_distance(p_u, e_hat_i.detach())  # sg(E_hat_I) — Eq. 9
        loss_i = _cos_distance(p_i, e_hat_u.detach())  # sg(E_hat_U) — Eq. 9

        return 0.5 * loss_u + 0.5 * loss_i  # Algorithm 2 outer loop

    def forward(self) -> Tuple[Tensor, Tensor]:
        """LightGCN propagation: layer-wise mean pooling of embeddings.

        Implements the standard LightGCN forward pass used as the backbone
        encoder f_theta in RecDCL (Section 4.2).

        Returns:
            Tuple[Tensor, Tensor]: (user_all_embeddings [n_users, F],
                                    item_all_embeddings [n_items+1, F]).
        """
        ego_embeddings = self.get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )

        # Ensure adjacency matrix is on the same device as embeddings
        if self.adj.device() != ego_embeddings.device:
            self.adj = self.adj.to(ego_embeddings.device)

        embeddings_list = [ego_embeddings]
        current_embeddings = ego_embeddings

        # L-layer message passing — LightGCN backbone (He et al. SIGIR 2020)
        for layer_module in self.propagation_network.children():
            current_embeddings = layer_module(current_embeddings, self.adj)
            embeddings_list.append(current_embeddings)

        # Mean pooling across layers 0 … L (equivalent to alpha=1/(L+1) weighting)
        all_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)

        # Split into user and item sub-matrices
        user_all_embeddings, item_all_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items + 1]
        )
        return user_all_embeddings, item_all_embeddings

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """One training iteration executing all three RecDCL loss components.

        Follows Algorithm 1 and Algorithm 2 from Appendix D.4:
          1. Encode via LightGCN to obtain E_U and E_I.
          2. L2-normalize embeddings (Algorithm 1 lines 4-5).
          3. Compute L_UIBT (Eq. 5), L_UUII (Eq. 6), L_BCL (Eq. 9).
          4. Combine with trade-off coefficients (Eq. 10).
          5. Update historical embedding buffers for the next step.

        Args:
            batch (Any): Triplet (user [B], pos_item [B], neg_item [B]).
                neg_item is discarded — RecDCL is a negative-sampling-free method.
            batch_idx (int): Current batch index.

        Returns:
            Tensor: Scalar training loss.
        """
        user, pos_item = batch

        # ---- Step 1: Encode ---- Algorithm 1 line 3
        user_all_emb, item_all_emb = self.forward()

        # Gather batch-level embeddings
        e_u = user_all_emb[user]  # [B, F] — E_U in the paper
        e_i = item_all_emb[pos_item]  # [B, F] — E_I in the paper

        # ---- Step 2: L2-normalize ---- Algorithm 1 lines 4-5
        # "Normalize e_u: e_u = e_u / ||e_u||"
        e_u = F.normalize(e_u, p=2, dim=-1)  # Algorithm 1 line 4
        e_i = F.normalize(e_i, p=2, dim=-1)  # Algorithm 1 line 5

        # ---- Step 3a: FCL — UIBT loss (Eq. 5) ---- Algorithm 1 line 6
        loss_uibt = self._uibt_loss(e_u, e_i, self.gamma)  # Eq. 5 / Algorithm 2

        # ---- Step 3b: FCL — UUII loss (Eq. 6) ---- Algorithm 1 line 7
        # Algorithm 2: L_UUII = UUII(e_u) / 2 + UUII(e_i) / 2
        loss_uuii = self._uuii_loss(
            e_u,
            e_i,
            poly_a=self.poly_a,
            poly_c=self.poly_c,
            poly_e=self.poly_e,
        )  # Eq. 6

        # ---- Step 3c: BCL loss (Eq. 9) ---- Algorithm 1 line 8
        # Historical embeddings are used to form the perturbed augmented view (Eq. 8).
        # On the very first step they are zeroed; we fall back to the current embeddings
        # as a warm-up (equivalent to tau_momentum=0 for the first iteration).
        if not self._hist_initialized:
            # ASSUMPTION: first-step fallback — use current embeddings as history
            # so Eq. 8 reduces to e_hat = current embedding.
            e_u_hist = e_u.detach()
            e_i_hist = e_i.detach()
        else:
            e_u_hist = self.hist_user_emb[user]  # type: ignore[index]
            e_i_hist = self.hist_item_emb[pos_item]  # type: ignore[index]

        # BCL uses the raw (non-normalized) embeddings for perturbed views.
        # ASSUMPTION: Eq. 8 operates on the propagated (possibly normalized) embeddings.
        # We apply normalization after blending for consistency with Algorithm 1.
        loss_bcl = self._bcl_loss(
            e_u,
            e_i,
            e_u_hist,
            e_i_hist,
            tau_momentum=self.tau_momentum,
            projector=self.projector,
        )  # Eq. 9 / Algorithm 2

        # ---- Step 4: Total loss (Eq. 10) ---- Algorithm 1 line 9
        # L = L_UIBT + alpha * L_UUII + beta * L_BCL
        loss = loss_uibt + self.alpha * loss_uuii + self.beta * loss_bcl  # Eq. 10

        # ---- Step 5: Update historical embedding buffers ----
        # Store the propagated (non-normalized) embeddings for the *next* step.
        # ASSUMPTION: historical embeddings are the full-graph propagated vectors
        # (not just the batch slice) to match the "target network" framing in
        # Section 4.2 ("online and target networks share the same graph encoder").
        # Storing only batch slices would be inconsistent across batches; storing
        # the full propagated matrix is more faithful to the momentum update style.
        with torch.no_grad():
            # SIMPLIFICATION: We update only the rows corresponding to users/items
            # seen in this batch to avoid storing a full [n_users+n_items, F] copy
            # every step, which would dominate memory for large catalogues.
            self.hist_user_emb[user] = user_all_emb[user].detach()  # type: ignore[operator]
            self.hist_item_emb[pos_item] = item_all_emb[pos_item].detach()  # type: ignore[operator]
            self._hist_initialized = True

        # ---- Logging ----
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("loss_uibt", loss_uibt, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss_uuii", loss_uuii, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss_bcl", loss_bcl, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute recommendation scores for the given users.

        At inference time, scores are computed as inner products between the
        propagated user and item embeddings.  No contrastive corrections are
        applied — the FCL/BCL objectives are purely training-time regularizers.

        Section 4.3: "we calculate the ranking score function by using the inner
        product between user and item representations."

        Args:
            user_indices (Tensor): Batch of user IDs [B].
            *args (Any): Unused positional arguments.
            item_indices (Optional[Tensor]): If None, scores against all items
                are returned [B, n_items].  Otherwise, scores for the sampled
                item sub-set [B, S].
            **kwargs (Any): Unused keyword arguments.

        Returns:
            Tensor: Score matrix [B, n_items] or [B, S].
        """
        user_all_embeddings, item_all_embeddings = self.propagate_embeddings()

        user_embeddings = user_all_embeddings[user_indices]  # [B, F]

        if item_indices is None:
            # Full ranking — score against all n_items (drop padding slot)
            item_embeddings = item_all_embeddings[:-1, :]  # [n_items, F]
            return torch.einsum("be,ie->bi", user_embeddings, item_embeddings)

        # Sampled ranking
        item_embeddings = item_all_embeddings[item_indices]  # [B, S, F]
        return torch.einsum("be,bse->bs", user_embeddings, item_embeddings)
