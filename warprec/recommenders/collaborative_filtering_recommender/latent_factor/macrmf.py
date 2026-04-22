# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="MACRMF")
class MACRMF(IterativeRecommender):
    """Implementation of MACR model
    from Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System (KDD 2021).

    - **Item module** Y_i(I): captures the direct effect of item popularity
      on the ranking score via path I → Y.
    - **User module** Y_u(U): captures user conformity (tendency to interact
      regardless of item-user match) via path U → Y.

    During **training** the three branches are fused multiplicatively (Eq. 7)
    and supervised jointly with a multi-task BCE loss (Eq. 8).

    During **inference** the counterfactual score (Eq. 9) subtracts the
    natural direct effect of item popularity, leaving only the total indirect
    effect through user-item matching (TIE = TE - NDE, Section 3.4).

    Args:
        params (dict): Model parameters.
        info (dict): Dataset information dictionary (must contain n_users, n_items).
        *args (Any): Variable length argument list.
        seed (int): Random seed for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: Pointwise (user, item, rating) loader with negative sampling.
        embedding_size (int): Dimension of user and item embeddings.
        alpha (float): Weight for item-module auxiliary loss L_I (Eq. 8).
        beta (float): Weight for user-module auxiliary loss L_U (Eq. 8).
        c (float): Reference matching score for counterfactual inference (Eq. 9).
        reg_weight (float): L2 regularization coefficient.
        batch_size (int): Training batch size.
        neg_samples (int): Number of negative samples per positive interaction.
        epochs (int): Number of training epochs.
        learning_rate (float): Adam learning rate.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.ITEM_RATING_LOADER

    # Model hyperparameters
    embedding_size: int
    alpha: float
    beta: float
    c: float
    reg_weight: float
    batch_size: int
    neg_samples: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(params, info, *args, seed=seed, **kwargs)

        # ------------------------------------------------------------------
        # Main branch — MF backbone (path U & I → K → Y)
        # "User-item matching: ŷ_k = Y_k(K(U=u, I=i))" — Section 3.3
        # ------------------------------------------------------------------
        # ASSUMPTION: dot-product inner product as matching function K(U, I),
        # consistent with standard MF and the paper's description of MF as
        # the backbone (Section 3.2, Appendix B).
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # ------------------------------------------------------------------
        # Item module — captures I → Y (item popularity direct effect)
        # "Item module: ŷ_i = Y_i(I=i)" — Section 3.3, Figure 5
        # SIMPLIFICATION: single linear projection from embedding to scalar,
        # rather than a deeper MLP. Paper states "can be implemented as MLPs"
        # but uses the simplest form; ablation (Table 3) shows even this
        # form yields substantial gains.
        # ------------------------------------------------------------------
        self.item_module = nn.Linear(self.embedding_size, 1)

        # ------------------------------------------------------------------
        # User module — captures U → Y (user conformity direct effect)
        # "User module: ŷ_u = Y_u(U=u)" — Section 3.3, Figure 5
        # SIMPLIFICATION: same single linear projection rationale as above.
        # ------------------------------------------------------------------
        self.user_module = nn.Linear(self.embedding_size, 1)

        self.apply(self._init_weights)
        self.reg_loss = EmbLoss()

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

    def forward(self, user: Tensor, item: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute per-branch scores and the fused training prediction.

        Implements the three-branch fusion:
            ŷ_ui = ŷ_k * σ(ŷ_i) * σ(ŷ_u)   — Eq. 7, Section 3.3

        Args:
            user (Tensor): User index tensor [batch_size].
            item (Tensor): Item index tensor [batch_size].

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                y_hat_k: Raw MF dot-product matching score [batch_size].
                y_hat_i: Raw item-module score [batch_size].
                y_hat_u: Raw user-module score [batch_size].
        """
        u_emb = self.user_embedding(user)  # [B, D]
        i_emb = self.item_embedding(item)  # [B, D]

        # Main branch: dot-product matching score (MF)
        # Section 3.2: "MF implements these functions as an element-wise
        # product … and a summation across embedding dimensions."
        y_hat_k = torch.mul(u_emb, i_emb).sum(dim=1)  # [B]  — Eq. matching score

        # Item module: popularity proxy score
        # Eq. in Section 3.3: ŷ_i = Y_i(I=i)
        y_hat_i = self.item_module(i_emb).squeeze(-1)  # [B]

        # User module: conformity proxy score
        # Eq. in Section 3.3: ŷ_u = Y_u(U=u)
        y_hat_u = self.user_module(u_emb).squeeze(-1)  # [B]

        return y_hat_k, y_hat_i, y_hat_u

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Single gradient step computing the MACR multi-task loss.

        Loss function (Eq. 8, Section 3.3):
            L = L_O + α * L_I + β * L_U

        where L_O, L_I, L_U are all binary cross-entropy losses.

        L_O supervises the fused prediction ŷ_ui = ŷ_k * σ(ŷ_i) * σ(ŷ_u).
        L_I supervises the item-module output σ(ŷ_i) alone.
        L_U supervises the user-module output σ(ŷ_u) alone.

        Args:
            batch (Any): Tuple of (user, item, rating) tensors.
            batch_idx (int): Batch index (unused).

        Returns:
            Tensor: The total scalar loss.
        """
        user, item, rating = batch  # rating ∈ {0.0, 1.0}
        rating = rating.float()

        y_hat_k, y_hat_i, y_hat_u = self.forward(user, item)

        # ------------------------------------------------------------------
        # Fused prediction for main task: ŷ_ui = ŷ_k * σ(ŷ_i) * σ(ŷ_u)
        # Eq. 7 — the overall ranking score used for L_O
        # ASSUMPTION: BCEWithLogitsLoss cannot be applied directly to ŷ_ui
        # because ŷ_ui is already a product of probabilities (not a raw logit).
        # We therefore apply BCE(σ(ŷ_ui_logit), y) as an approximation, but
        # the paper uses plain BCE on the product:
        #   L_O = Σ -y*log(ŷ_ui) - (1-y)*log(1-ŷ_ui)   — Eq. 6
        # We compute ŷ_ui in [0,1] space and apply binary_cross_entropy.
        # ------------------------------------------------------------------
        sig_i = torch.sigmoid(y_hat_i)  # σ(ŷ_i) ∈ (0, 1)
        sig_u = torch.sigmoid(y_hat_u)  # σ(ŷ_u) ∈ (0, 1)

        # Fused prediction — Eq. 7: ŷ_ui = ŷ_k * σ(ŷ_i) * σ(ŷ_u)
        # ASSUMPTION: ŷ_k (raw dot product) is used directly as the matching
        # score without further sigmoid, consistent with the paper's description
        # and the fact that the auxiliary losses on σ(ŷ_i) and σ(ŷ_u) already
        # bound the probability-like fused output implicitly.
        y_hat_ui = y_hat_k * sig_i * sig_u  # Eq. 7 — fused score [B]

        # Main recommendation loss L_O — Eq. 6 (BCE over fused prediction)
        loss_o = F.binary_cross_entropy_with_logits(y_hat_ui, rating)

        # Item auxiliary loss L_I — Eq. 8 bottom: BCE over σ(ŷ_i)
        # "L_I = Σ -y*log(σ(ŷ_i)) - (1-y)*log(1-σ(ŷ_i))"  — Section 3.3
        loss_i = F.binary_cross_entropy_with_logits(y_hat_i, rating)

        # User auxiliary loss L_U — Eq. 8 top: BCE over σ(ŷ_u)
        # "L_U = Σ -y*log(σ(ŷ_u)) - (1-y)*log(1-σ(ŷ_u))"  — Section 3.3
        loss_u = F.binary_cross_entropy_with_logits(y_hat_u, rating)

        # Combined multi-task loss — Eq. 8: L = L_O + α*L_I + β*L_U
        loss_main = loss_o + self.alpha * loss_i + self.beta * loss_u

        # L2 regularization on embeddings — Appendix B (coefficient 1e-5 default)
        loss_reg = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(item),
        )

        loss = loss_main + loss_reg
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
        """Counterfactual debiased prediction (TIE-based ranking).

        Implements Eq. 9 from Section 3.3:
            ŷ_cf = ŷ_k * σ(ŷ_i) * σ(ŷ_u) − c * σ(ŷ_i) * σ(ŷ_u)

        This is the Total Indirect Effect (TIE = TE − NDE) which removes the
        Natural Direct Effect of item popularity from the ranking score,
        leaving only the user–item matching contribution (Section 3.4).

        The hyperparameter `c` represents the reference status of ŷ_k when
        user–item matching is blocked (i.e. K = K_{u*,i*}), typically the
        mean matching score at reference embeddings (Section 3.4, Eq. 10).

        Args:
            user_indices (Tensor): Batch of user indices [batch_size].
            *args (Any): Unused positional arguments.
            item_indices (Optional[Tensor]): Item indices for sampled evaluation
                [batch_size, pad_seq]. If None, full ranking over all items.
            **kwargs (Any): Unused keyword arguments.

        Returns:
            Tensor: Debiased ranking scores [batch_size, n_items] or
                [batch_size, pad_seq].
        """
        u_emb = self.user_embedding(user_indices)  # [B, D]

        # User-module score for the batch
        y_hat_u = self.user_module(u_emb).squeeze(-1)  # [B]
        sig_u = torch.sigmoid(y_hat_u)  # [B]

        if item_indices is None:
            # Full ranking — score against all n_items
            i_emb = self.item_embedding.weight[:-1, :]  # [n_items, D]

            # MF dot-product: ŷ_k [B, n_items]
            y_hat_k = torch.einsum("be,ie->bi", u_emb, i_emb)

            # Item popularity scores [n_items]
            y_hat_i_all = self.item_module(i_emb).squeeze(-1)  # [n_items]
            sig_i = torch.sigmoid(y_hat_i_all).unsqueeze(0)  # [1, n_items]
            sig_u_2d = sig_u.unsqueeze(1)  # [B, 1]

        else:
            # Sampled ranking — score against a provided subset of items
            i_emb = self.item_embedding(item_indices)  # [B, S, D]

            # MF dot-product: ŷ_k [B, S]
            y_hat_k = torch.einsum("be,bse->bs", u_emb, i_emb)

            # Item popularity scores [B, S]
            y_hat_i_all = self.item_module(i_emb).squeeze(-1)  # [B, S]
            sig_i = torch.sigmoid(y_hat_i_all)  # [B, S]
            sig_u_2d = sig_u.unsqueeze(1)  # [B, 1]

        # Counterfactual score — Eq. 9:
        #   ŷ_cf = ŷ_k * σ(ŷ_i) * σ(ŷ_u) − c * σ(ŷ_i) * σ(ŷ_u)
        #        = (ŷ_k − c) * σ(ŷ_i) * σ(ŷ_u)
        # This equals TIE = TE − NDE (Section 3.4, Eq. 10).
        scores = (y_hat_k - self.c) * sig_i * sig_u_2d  # [B, n_items] or [B, S]
        return scores
