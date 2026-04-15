# pylint: disable = R0801, R0901, R0902, R0914, R0915
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix, csr_matrix
from torch import Tensor, nn
from torch.utils.data import DataLoader

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class GCNTeacher(nn.Module):
    """Lightweight GCN teacher model.

    Implements the teacher propagation described in Eq 4:
        H^(t) = sum_{l=0..L} H^(t)_l,   H^(t)_{l+1} = D^{-1/2}(A_bar+I)D^{-1/2} * H^(t)_l

    # Eq 4 — LightGCN-style sum-of-layer propagation with symmetric normalization
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_size: int,
        n_layers: int,
        adj_norm: Tensor,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        # Eq 4 — initial embedding H^(t)_0 = H_bar^(t)
        self.user_embedding = nn.Embedding(n_users, embedding_size)
        self.item_embedding = nn.Embedding(
            n_items + 1, embedding_size, padding_idx=n_items
        )

        # ASSUMPTION: adj_norm is the symmetric-normalized adjacency  D^{-1/2}(A_bar+I)D^{-1/2}
        # stored as a dense [n_users+n_items+1, n_users+n_items+1] sparse float tensor.
        # We register it as a buffer so it moves with the model.
        self.register_buffer("adj_norm", adj_norm)

    def forward(self) -> Tuple[Tensor, list]:
        """Perform L-layer graph propagation and return per-layer user embeddings.

        Returns:
            Tuple[Tensor, list]: Summed embedding and list of L+1 per-layer embedding
                matrices (each is stacked [user|item] tensor). The caller uses
                layers [2..L] for embedding-level distillation (Eq 8).
        """
        # Eq 4 — initialise from raw embeddings
        ego = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )  # [(n_users + n_items + 1), d]

        layer_embeddings = [ego]
        h = ego
        for _ in range(self.n_layers):
            # Eq 4 — H^(t)_{l+1} = A_norm * H^(t)_l
            h = torch.sparse.mm(self.adj_norm, h)  # pylint: disable=not-callable
            layer_embeddings.append(h)

        # Eq 4 — sum over all layers: H^(t) = sum_{l=0..L} H^(t)_l
        h_sum = torch.stack(layer_embeddings, dim=0).sum(dim=0)

        # Return the summed embedding AND all individual layer embeddings
        # (individual layers needed for L2, Eq 8)
        return h_sum, layer_embeddings

    def get_user_item_embeddings(self) -> Tuple[Tensor, Tensor]:
        """Return final user and item embeddings (used during pre-training predict)."""
        h_sum, _ = self.forward()
        user_emb = h_sum[: self.n_users]
        item_emb = h_sum[self.n_users : self.n_users + self.n_items]
        return user_emb, item_emb


class MLPStudent(nn.Module):
    """MLP student network.

    Implements the student embedding function from Eq 5:
        h^(s)_i = FC_{L'}(h_bar^(s)_i),
        FC(h_bar) = delta(W h_bar) + h_bar          # residual connection

    # Eq 5 — shared FC stack with LeakyReLU and residual connections
    """

    def __init__(self, embedding_size: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(embedding_size, embedding_size) for _ in range(n_layers)]
        )
        # Eq 5 — LeakyReLU activation delta(·)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: Tensor) -> Tensor:
        """Apply L' FC layers with residual connections.

        Args:
            x (Tensor): Raw embedding vectors, shape [batch, d].

        Returns:
            Tensor: Refined embedding, shape [batch, d].
        """
        for layer in self.layers:
            # Eq 5 — FC(h_bar) = delta(W h_bar) + h_bar
            x = self.activation(layer(x)) + x
        return x


@model_registry.register(name="SimRec")
class SimRec(IterativeRecommender):
    """Implementation of SimRec from
    Graph-less Collaborative Filtering via Contrastive Knowledge Distillation (WWW 2023)

    SimRec distills knowledge from a GCN teacher into a lightweight MLP student
    using prediction-level KD (L1), embedding-level contrastive KD (L2), and
    adaptive contrastive regularization (L3) to address over-smoothing.

    Args:
        params (dict): Model hyperparameters.
        info (dict): Dataset metadata (n_users, n_items).
        interactions (Interactions): Training user-item interactions.
        *args (Any): Variable length argument list.
        seed (int): Random seed.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: Uses POS_NEG_LOADER for BPR-style contrastive sampling of T2.
        embedding_size (int): Dimensionality of embedding vectors.
        n_teacher_layers (int): GCN propagation layers.
        n_student_layers (int): MLP FC layers.
        teacher_reg_weight (float): L2 weight for teacher pre-training.
        lambda1 (float): Weight for prediction-level distillation loss L1.
        lambda2 (float): Weight for embedding-level distillation loss L2.
        lambda3 (float): Weight for adaptive contrastive regularization L3.
        lambda4 (float): Weight for MLP weight-decay L4.
        tau1 (float): Temperature for prediction-level distillation.
        tau2 (float): Temperature for embedding-level distillation.
        tau3 (float): Temperature for adaptive contrastive regularization.
        eps (float): Epsilon for adaptive weight adjustment.
        batch_size_kd (int): Number of KD samples |T1| per step.
        teacher_epochs (int): Epochs to pre-train the GCN teacher.
        batch_size (int): Mini-batch size |T2|.
        epochs (int): Number of student training epochs.
        learning_rate (float): Adam LR for student.
        teacher_learning_rate (float): Adam LR for teacher.
    """

    # Dataloader definition
    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    # Model hyperparameters
    embedding_size: int
    n_teacher_layers: int
    n_student_layers: int
    teacher_reg_weight: float
    lambda1: float
    lambda2: float
    lambda3: float
    lambda4: float
    tau1: float
    tau2: float
    tau3: float
    eps: float
    batch_size_kd: int
    teacher_epochs: int
    batch_size: int
    epochs: int
    learning_rate: float
    teacher_learning_rate: float

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

        # ---- Build normalized adjacency matrix (D^{-1/2} (A_bar + I) D^{-1/2}) ----
        adj_norm = self._build_norm_adj(
            interactions.get_sparse().tocoo(), self.n_users, self.n_items
        )

        # ---- Teacher (frozen after pre-training) ----
        self.teacher = GCNTeacher(
            n_users=self.n_users,
            n_items=self.n_items,
            embedding_size=self.embedding_size,
            n_layers=self.n_teacher_layers,
            adj_norm=adj_norm,
        )

        # ---- Student (trained with KD) ----
        # Eq 5 — initial embeddings h_bar^(s)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )
        # Eq 5 — shared MLP network
        self.mlp = MLPStudent(self.embedding_size, self.n_student_layers)

        # ---- Losses ----
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # ---- Pre-training state ----
        # ASSUMPTION: teacher pre-training is done in the first `teacher_epochs` epochs
        # using the same training_step method with a phase flag.
        self._teacher_pretrained: bool = False

        # ---- Teacher optimizer (separate from Lightning-managed student optimizer) ----
        self._teacher_optimizer: Optional[torch.optim.Adam] = None

        # ---- Weight initialization ----
        self.apply(self._init_weights)

    @staticmethod
    def _build_norm_adj(
        interaction_matrix: coo_matrix, n_users: int, n_items: int
    ) -> Tensor:
        """Build symmetric-normalized adjacency matrix D^{-1/2}(A_bar+I)D^{-1/2}.

        # Eq 4 — symmetric normalization used in the GCN teacher propagation

        Args:
            interaction_matrix (coo_matrix): User-item interactions in COO format.
            n_users (int): Number of users.
            n_items (int): Number of items (excluding padding).

        Returns:
            Tensor: Sparse float tensor of shape [n_users + n_items + 1, n_users + n_items + 1].
        """
        # SIMPLIFICATION: We include a padding node (item index n_items) so that the
        # adjacency dimension aligns with the item embedding which uses padding_idx=n_items.
        total = n_users + n_items + 1

        user_nodes = interaction_matrix.row.astype(np.int64)
        item_nodes = (interaction_matrix.col + n_users).astype(np.int64)

        # Build bipartite + identity edges (A_bar + I)
        row = np.concatenate([user_nodes, item_nodes, np.arange(total, dtype=np.int64)])
        col = np.concatenate([item_nodes, user_nodes, np.arange(total, dtype=np.int64)])
        data = np.ones(len(row), dtype=np.float32)

        # Aggregate duplicates via CSR
        adj_csr = csr_matrix((data, (row, col)), shape=(total, total))
        adj_coo = adj_csr.tocoo()

        rows_t = torch.from_numpy(adj_coo.row.astype(np.int64))
        cols_t = torch.from_numpy(adj_coo.col.astype(np.int64))
        vals_t = torch.from_numpy(adj_coo.data.astype(np.float32))

        # Degree vector for symmetric normalization
        deg = torch.zeros(total, dtype=torch.float32)
        deg.scatter_add_(0, rows_t, vals_t)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        # D^{-1/2} * A * D^{-1/2}
        norm_vals = deg_inv_sqrt[rows_t] * vals_t * deg_inv_sqrt[cols_t]

        adj_norm = torch.sparse_coo_tensor(
            torch.stack([rows_t, cols_t], dim=0),
            norm_vals,
            (total, total),
        ).coalesce()
        return adj_norm

    def get_dataloader(
        self, interactions: Interactions, sessions: Sessions, **kwargs: Any
    ) -> DataLoader:
        return interactions.get_contrastive_dataloader(
            batch_size=self.batch_size, **kwargs
        )

    def _pretrain_teacher_step(
        self, user: Tensor, pos_item: Tensor, neg_item: Tensor
    ) -> Tensor:
        """One BPR gradient step on the GCN teacher.

        # Eq 11 — L^(t) = -sum log sigma(y^(t)_{i,j} - y^(t)_{i,k}) + lambda^(t) ||H_bar^(t)||^2_F
        """
        if self._teacher_optimizer is None:
            self._teacher_optimizer = torch.optim.Adam(
                self.teacher.parameters(), lr=self.teacher_learning_rate
            )

        self._teacher_optimizer.zero_grad()

        user_emb, item_emb = self.teacher.get_user_item_embeddings()
        u = user_emb[user]  # [B, d]
        pos = item_emb[pos_item]  # [B, d]
        neg = item_emb[neg_item]  # [B, d]

        pos_scores = (u * pos).sum(dim=-1)
        neg_scores = (u * neg).sum(dim=-1)

        # Eq 11 — BPR loss
        bpr = self.bpr_loss(pos_scores, neg_scores)

        # Eq 11 — weight-decay on initial embeddings
        reg = self.teacher_reg_weight * self.reg_loss(
            self.teacher.user_embedding(user),
            self.teacher.item_embedding(pos_item),
            self.teacher.item_embedding(neg_item),
        )
        loss = bpr + reg
        loss.backward()
        self._teacher_optimizer.step()
        return loss.detach()

    def forward(self, user_indices: Tensor) -> Tensor:
        """Apply MLP student to user embeddings.

        # Eq 5 — h^(s)_i = FC_{L'}(h_bar^(s)_i)

        This signature satisfies the Lightning/WarpRec abstract requirement.
        For item encoding use _encode_item; training_step calls both directly.

        Args:
            user_indices (Tensor): User indices, shape [B].

        Returns:
            Tensor: Refined user embeddings, shape [B, d].
        """
        return self._encode_user(user_indices)

    def _encode_user(self, user_idx: Tensor) -> Tensor:
        """Eq 5 — student embedding for users."""
        return self.mlp(self.user_embedding(user_idx))

    def _encode_item(self, item_idx: Tensor) -> Tensor:
        """Eq 5 — student embedding for items."""
        return self.mlp(self.item_embedding(item_idx))

    def _loss_l1(
        self, user_s: Tensor, teacher_user_emb: Tensor, teacher_item_emb: Tensor
    ) -> Tensor:
        """Prediction-level knowledge distillation loss.

        Samples |T1| random (u_i, v_j, v_k) triplets where v_j, v_k are drawn
        uniformly from ALL items (not just positive/negative).

        # Eq 6 — z_{i,j,k} = h_i^T h_j - h_i^T h_k
        # Eq 7 — L1 = sum -[z_bar^(t) log z_bar^(s) + (1-z_bar^(t)) log(1-z_bar^(s))]
        #             z_bar = sigmoid(z / tau1)

        # SIMPLIFICATION: T1 is a random sub-batch of size min(batch_size_kd, n_items^2)
        #   drawn by sampling random item indices rather than building all pairs.
        #   This keeps memory bounded without changing the expectation of the gradient.
        """
        B = user_s.size(0)
        device = user_s.device

        # Sample random item pairs (v_j, v_k) from all items
        kd_size = min(self.batch_size_kd, B * 16)
        j_idx = torch.randint(0, self.n_items, (kd_size,), device=device)
        k_idx = torch.randint(0, self.n_items, (kd_size,), device=device)

        # User indices aligned with item samples (cycle over the batch)
        u_idx = torch.arange(kd_size, device=device) % B

        u_s = user_s[u_idx]  # [kd_size, d]
        h_j_t = teacher_item_emb[j_idx]  # [kd_size, d]
        h_k_t = teacher_item_emb[k_idx]

        # Eq 6 — difference scores for teacher
        # ASSUMPTION: teacher embeddings are from the final summed H^(t), not per-layer
        h_j_s = self._encode_item(j_idx)  # student item embeddings
        h_k_s = self._encode_item(k_idx)

        u_t = teacher_user_emb[torch.arange(B, device=device)[u_idx % B]]

        z_s = (u_s * h_j_s).sum(-1) - (u_s * h_k_s).sum(-1)  # Eq 6 — student
        z_t = (u_t * h_j_t).sum(-1) - (u_t * h_k_t).sum(-1)  # Eq 6 — teacher (no grad)
        z_t = z_t.detach()

        # Eq 7 — soft labels via sigmoid with temperature
        z_bar_t = torch.sigmoid(z_t / self.tau1)
        z_bar_s = torch.sigmoid(z_s / self.tau1)

        # Eq 7 — binary cross-entropy between soft teacher and student labels
        loss = F.binary_cross_entropy(z_bar_s.clamp(1e-7, 1 - 1e-7), z_bar_t.detach())
        return loss

    def _loss_l2(
        self,
        user_s: Tensor,
        item_s: Tensor,
        user_idx: Tensor,
        item_idx: Tensor,
        teacher_layer_embeddings: list,
    ) -> Tensor:
        """Embedding-level contrastive knowledge distillation.

        # Eq 8 — InfoNCE between student embeddings and sum of high-order teacher layers (l=2..L)

        # ASSUMPTION: "high-order" is defined as layers l>=2 as per the paper's Eq 8 notation.
        #   If n_teacher_layers < 2, we fall back to using all available teacher layers.
        """
        # Eq 8 — high-order teacher embeddings: sum of layers l=2..L
        n_layers = len(teacher_layer_embeddings) - 1  # 0-indexed; last is layer L
        start = min(2, n_layers)  # fallback if fewer than 2 layers

        # Sum layers [start..L] for users and items separately
        # teacher_layer_embeddings[l] has shape [(n_users + n_items + 1), d]
        high_order_layers = teacher_layer_embeddings[start:]
        if len(high_order_layers) == 0:
            high_order_layers = teacher_layer_embeddings  # fallback
        high_order = (
            torch.stack(high_order_layers, dim=0).sum(dim=0).detach()
        )  # no grad through teacher

        t_user_high = high_order[user_idx]  # [B, d] high-order teacher user embeddings
        t_item_high = high_order[
            self.n_users + item_idx
        ]  # [B, d] high-order teacher item embeddings

        # Eq 8 — InfoNCE loss over users
        # numerator: cos(h^(s)_i, sum_{l>=2} h^(t)_{i,l}) / tau2
        def _infonce(
            anchors: Tensor, positives: Tensor, all_negatives: Tensor
        ) -> Tensor:
            """InfoNCE: -log( exp(cos(a,p)/tau) / sum_j exp(cos(a,n_j)/tau) )."""
            anchors_n = F.normalize(anchors, dim=-1)
            positives_n = F.normalize(positives, dim=-1)
            all_n = F.normalize(all_negatives, dim=-1)

            pos_sim = (anchors_n * positives_n).sum(
                -1, keepdim=True
            ) / self.tau2  # [B, 1]
            all_sim = torch.matmul(anchors_n, all_n.T) / self.tau2  # [B, N]

            # Eq 8 — numerator exp / denominator sum exp
            loss = -pos_sim.squeeze(1) + torch.logsumexp(all_sim, dim=-1)
            return loss.mean()

        l2_user = _infonce(user_s, t_user_high, t_user_high)
        l2_item = _infonce(item_s, t_item_high, t_item_high)
        return l2_user + l2_item

    def _compute_adaptive_weight(
        self,
        user_s: Tensor,
        item_s: Tensor,
        rec_loss: Tensor,
        l1_loss: Tensor,
        l2_loss: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute per-node adaptive contrastive regularization weights omega_i, omega_j.

        # Eq 10 — omega_i = 1-eps if grad_{1,2}^T grad_{rec} > grad_1^T grad_2, else 1+eps

        # SIMPLIFICATION: Computing per-sample exact gradient vectors is expensive.
        #   We approximate by computing scalar gradient norms via autograd for the
        #   current batch, rather than per-node gradient vectors as in the paper.
        #   The sign of the dot-products is approximated via the sign of
        #   (g_12 . g_rec > g_1 . g_2) evaluated at the batch level.
        """
        # Retain graph so we can differentiate multiple times
        g_12 = torch.autograd.grad(
            l1_loss + l2_loss,
            [user_s, item_s],
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        g_rec = torch.autograd.grad(
            rec_loss,
            [user_s, item_s],
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        g_1 = torch.autograd.grad(
            l1_loss,
            [user_s, item_s],
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        g_2 = torch.autograd.grad(
            l2_loss,
            [user_s, item_s],
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )

        # Helper to safely handle None gradients (e.g., l1_loss doesn't depend on item_s)
        def _safe_grad(
            grad_tuple: Tuple[Optional[Tensor], ...], idx: int, ref_tensor: Tensor
        ) -> Tensor:
            g = grad_tuple[idx]
            return g if g is not None else torch.zeros_like(ref_tensor)

        g_12_u = _safe_grad(g_12, 0, user_s)
        g_12_i = _safe_grad(g_12, 1, item_s)

        g_rec_u = _safe_grad(g_rec, 0, user_s)
        g_rec_i = _safe_grad(g_rec, 1, item_s)

        g_1_u = _safe_grad(g_1, 0, user_s)
        g_1_i = _safe_grad(g_1, 1, item_s)

        g_2_u = _safe_grad(g_2, 0, user_s)
        g_2_i = _safe_grad(g_2, 1, item_s)

        # Eq 10 — compare dot products to decide weight direction
        # Users (index 0)
        dot_12_rec_u = (g_12_u * g_rec_u).sum(dim=-1)  # Shape: [B]
        dot_1_2_u = (g_1_u * g_2_u).sum(dim=-1)  # Shape: [B]

        # Items (index 1)
        dot_12_rec_i = (g_12_i * g_rec_i).sum(dim=-1)  # Shape: [B]
        dot_1_2_i = (g_1_i * g_2_i).sum(dim=-1)  # Shape: [B]

        # Per-node conditions (Shape: [B, 1] so it broadcasts with losses later)
        cond_u = (dot_12_rec_u > dot_1_2_u).float().unsqueeze(1)
        omega_u = 1.0 - self.eps * cond_u + self.eps * (1.0 - cond_u)

        cond_i = (dot_12_rec_i > dot_1_2_i).float().unsqueeze(1)
        omega_i = 1.0 - self.eps * cond_i + self.eps * (1.0 - cond_i)

        return omega_u, omega_i

    def _loss_l3(
        self,
        user_s: Tensor,
        item_s: Tensor,
        omega_u: Tensor,
        omega_i: Tensor,
    ) -> Tensor:
        """Adaptive contrastive regularization pushing all node embeddings apart.

        # Eq 9 — L3 = sum_u [phi(u,U,w_u) + phi(u,V,w_u)] + sum_v phi(v,V,w_v)
        #   phi(u_i, U, w_i) = w_i * log sum_{u'} exp(h^(s)_i^T h^(s)_{i'} / tau3)

        # SIMPLIFICATION: All-pairs similarity within the batch is used as an
        #   approximation of the full-population sum in Eq 9. The paper sums over
        #   the full U and V; using batch-level negatives is standard practice
        #   and avoids O(N^2) complexity.
        """
        # Ensure omega is broadcastable to [B]
        omega_u = omega_u.squeeze(-1)
        omega_i = omega_i.squeeze(-1)

        # Eq 9 — user-user push-away
        uu_sim = torch.matmul(user_s, user_s.T) / self.tau3  # [B, B]
        uu_sim.fill_diagonal_(-float("inf"))  # exclude self-similarity
        l3_uu = (omega_u * torch.logsumexp(uu_sim, dim=-1)).mean()

        # Eq 9 — user-item push-away
        ui_sim = torch.matmul(user_s, item_s.T) / self.tau3  # [B, B]
        l3_ui = (omega_u * torch.logsumexp(ui_sim, dim=-1)).mean()

        # Eq 9 — item-item push-away
        ii_sim = torch.matmul(item_s, item_s.T) / self.tau3  # [B, B]
        ii_sim.fill_diagonal_(-float("inf"))  # exclude self-similarity
        l3_ii = (omega_i * torch.logsumexp(ii_sim, dim=-1)).mean()

        return l3_uu + l3_ui + l3_ii

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Combined teacher pre-training and student KD training.

        Phase 1 (epochs 0..teacher_epochs-1): Only update the GCN teacher via BPR.
        Phase 2 (epochs teacher_epochs..total): Freeze teacher, update student with
            the full SimRec objective L^(s) = Lrec + lambda1*L1 + lambda2*L2 + lambda3*L3 + lambda4*L4.

        # Algorithm 1 — SimRec learning procedure
        # Eq 12 — L^(s) = Lrec + lambda1*L1 + lambda2*L2 + lambda3*L3 + lambda4*L4
        """
        user, pos_item, neg_item = batch

        current_epoch = self.current_epoch  # provided by Lightning

        # ---- Phase 1: Teacher pre-training ----
        if current_epoch < self.teacher_epochs:
            # Algorithm 1, line 2 — Train GCN teacher until convergence
            teacher_loss = self._pretrain_teacher_step(user, pos_item, neg_item)
            self.log(
                "teacher_loss",
                teacher_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            # Return a zero-grad placeholder loss to satisfy Lightning (student params untouched)
            # SIMPLIFICATION: Return a zero tensor with grad to keep Lightning happy.
            dummy: Tensor = torch.tensor(0.0, device=user.device, requires_grad=True)
            for p in self.mlp.parameters():
                dummy = dummy + p.sum() * 0
            for p in self.user_embedding.parameters():
                dummy = dummy + p.sum() * 0
            return dummy

        # ---- Phase 2: Student KD training (teacher frozen) ----
        # Mark teacher as pre-trained and freeze its parameters
        if not self._teacher_pretrained:
            for param in self.teacher.parameters():
                param.requires_grad_(False)
            self._teacher_pretrained = True

        # Algorithm 1, line 4 — mini-batch T2 drawn from E
        # T2 = {(u_i, v_j)} from the batch (we have user + pos_item)
        user_idx = user
        item_idx = pos_item

        # Student embeddings (requires_grad=True for adaptive weight computation)
        user_s = self._encode_user(user_idx)  # h^(s)_i, Eq 5
        item_s = self._encode_item(item_idx)  # h^(s)_j, Eq 5

        # Retain grad for adaptive weight computation (Eq 10)
        user_s.retain_grad()
        item_s.retain_grad()

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_sum, teacher_layers = self.teacher()

        teacher_user_emb = teacher_sum[: self.n_users]
        teacher_item_emb = teacher_sum[self.n_users : self.n_users + self.n_items]

        # Eq 12 — Lrec: recommendation objective (positive pair similarity)
        # Lrec = -sum_{(u_i,v_j) in T2} y_{i,j}  (maximize positive scores)
        rec_scores = (user_s * item_s).sum(dim=-1)
        l_rec = -rec_scores.mean()

        # Algorithm 1, line 5 — sample T1 for prediction-level distillation
        # Eq 7 — L1 prediction-level KD
        l1 = self._loss_l1(user_s, teacher_user_emb, teacher_item_emb)

        # Algorithm 1, line 8 — Eq 8 embedding-level KD
        l2 = self._loss_l2(user_s, item_s, user_idx, item_idx, teacher_layers)

        # Algorithm 1, line 9 — Eq 10 adaptive weight computation
        # We compute omega using retained gradients from l_rec, l1, l2
        try:
            omega_u, omega_i = self._compute_adaptive_weight(
                user_s, item_s, l_rec, l1, l2
            )
        except (RuntimeError, ValueError):
            # ASSUMPTION: If gradient graph is broken (e.g., first step), fall back to omega=1
            omega_u = torch.ones(user_s.size(0), 1, device=user_s.device)
            omega_i = torch.ones(item_s.size(0), 1, device=item_s.device)

        # Algorithm 1, line 10 — Eq 9 adaptive contrastive regularization
        l3 = self._loss_l3(user_s, item_s, omega_u, omega_i)

        # Eq 12 — L4 weight-decay on student embedding tables
        l4 = self.reg_loss(
            self.user_embedding(user_idx),
            self.item_embedding(item_idx),
        )

        # Eq 12 — full student objective
        loss = (
            l_rec
            + self.lambda1 * l1
            + self.lambda2 * l2
            + self.lambda3 * l3
            + self.lambda4 * l4
        )

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("l_rec", l_rec.detach(), on_step=False, on_epoch=True)
        self.log("l1", l1.detach(), on_step=False, on_epoch=True)
        self.log("l2", l2.detach(), on_step=False, on_epoch=True)
        self.log("l3", l3.detach(), on_step=False, on_epoch=True)
        return loss

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Score users against items using the trained MLP student.

        # Eq 2 — y_{i,j} = h_i^T h_j,  h_i = M-Embed(h_bar_i)

        Args:
            user_indices (Tensor): User indices, shape [batch_size].
            *args (Any): Variable length argument list.
            item_indices (Optional[Tensor]): Optional item indices, shape [batch_size, k].
                If None, scores all items.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Tensor: Score tensor, shape [batch_size, n_items] or [batch_size, k].
        """
        user_emb = self._encode_user(user_indices)  # [B, d]

        if item_indices is None:
            # Full prediction: score against all items
            all_item_emb = self.mlp(
                self.item_embedding.weight[: self.n_items]
            )  # [n_items, d]
            return torch.matmul(user_emb, all_item_emb.T)  # [B, n_items]

        # Sampled prediction
        item_emb = self.mlp(self.item_embedding(item_indices))  # [B, k, d]
        return torch.einsum("be,bse->bs", user_emb, item_emb)  # [B, k]
