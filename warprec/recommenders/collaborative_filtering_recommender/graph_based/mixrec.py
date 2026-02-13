# pylint: disable = R0801, E1102
from typing import Tuple, Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import LGConv

from warprec.data.entities import Interactions, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="MixRec")
class MixRec(IterativeRecommender, GraphRecommenderUtils):
    """Implementation of MixRec algorithm from
    "MixRec: Individual and Collective Mixing Empowers Data Augmentation for Recommender Systems" (WWW '25).

    For further details, check the `paper <https://doi.org/10.1145/3696410.3714565>`_.

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
        n_layers (int): The number of LightGCN propagation layers.
        ssl_lambda (float): Weight for Contrastive Loss.
        alpha (float): Shape parameter for Beta distribution (for Individual Mixing).
        temperature (float): Temperature for InfoNCE.
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
    ssl_lambda: float
    alpha: float
    temperature: float
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

        # Graph Construction (LightGCN Encoder)
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,
            normalize=True,
        )

        # Propagation Network
        self.propagation_network = nn.ModuleList(
            [LGConv() for _ in range(self.n_layers)]
        )

        # Vectorized normalization for embedding aggregation (Mean pooling)
        alpha_tensor = torch.full(
            (self.n_layers + 1,), 1.0 / (self.n_layers + 1), device=self.device
        )
        self.register_buffer("alpha_gcn", alpha_tensor)

        # Initialize weights
        self.apply(self._init_weights)

        # Losses
        self.bpr_loss = BPRLoss()
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
        """Standard LightGCN Forward pass to get encoded embeddings."""
        ego_embeddings = self.get_ego_embeddings(
            self.user_embedding, self.item_embedding
        )

        if self.adj.device() != ego_embeddings.device:
            self.adj = self.adj.to(ego_embeddings.device)

        embeddings_list = [ego_embeddings]
        current_embeddings = ego_embeddings

        for conv_layer in self.propagation_network:
            current_embeddings = conv_layer(current_embeddings, self.adj)
            embeddings_list.append(current_embeddings)

        # Weighted sum
        final_embeddings = torch.zeros_like(ego_embeddings)
        for k, emb in enumerate(embeddings_list):
            final_embeddings += emb * self.alpha_gcn[k]  # type: ignore[index]

        user_final, item_final = torch.split(
            final_embeddings, [self.n_users, self.n_items + 1]
        )
        return user_final, item_final

    def _mix_embeddings(
        self, original: Tensor, shuffled: Tensor, beta: Tensor
    ) -> Tensor:
        """Individual Mixing: Linear interpolation."""
        return beta * original + (1 - beta) * shuffled

    def _collective_mixing(self, embeddings: Tensor) -> Tensor:
        """Collective Mixing.

        Generates new samples by forming convex combinations of the entire batch.
        To be efficient and avoid O(B^2) sampling, we sample one set of Dirichlet
        weights per batch (or a small number of sets) and broadcast.
        """
        batch_size = embeddings.size(0)
        # Sample coefficients from Dirichlet(1, 1, ..., 1)
        dir_dist = torch.distributions.Dirichlet(
            torch.ones(batch_size, device=self.device)
        )  # [1, Batch]
        coeffs = dir_dist.sample().unsqueeze(0)

        # Weighted sum of all embeddings in batch: [1, Batch] x [Batch, Dim] -> [1, Dim]
        collective_view = torch.mm(coeffs, embeddings)

        # Expand to match batch size for loss calculation
        return collective_view.expand(batch_size, -1)

    def _hard_nce_loss(
        self,
        anchor: Tensor,
        positive: Tensor,
        neg_disorder: Tensor,
        neg_collective: Tensor,
        temperature: float,
    ) -> Tensor:
        """Computes InfoNCE loss with hard negatives."""
        # L2 normalization
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        neg_disorder = F.normalize(neg_disorder, p=2, dim=1)
        neg_collective = F.normalize(neg_collective, p=2, dim=1)

        # Positive similarity (Anchor vs Mixed)
        pos_sim = (anchor * positive).sum(dim=1) / temperature  # [Batch]

        # Hard negative 1 similarity (Anchor vs Disorder)
        dis_sim = (anchor * neg_disorder).sum(dim=1) / temperature  # [Batch]

        # Hard negative 2 similarity (Anchor vs Collective)
        col_sim = (anchor * neg_collective).sum(dim=1) / temperature  # [Batch]

        # Batch negatives similarity (Anchor vs Mixed)
        batch_sim_matrix = (
            torch.mm(anchor, positive.t()) / temperature
        )  # [Batch, Batch]

        all_logits = torch.cat(
            [batch_sim_matrix, dis_sim.unsqueeze(1), col_sim.unsqueeze(1)], dim=1
        )  # [B, B + 2]

        # Loss = -log( exp(pos) / sum(exp(all)) )
        #      = -pos + logsumexp(all)

        loss = -pos_sim + torch.logsumexp(all_logits, dim=1)
        return loss

    def _dual_mixing_cl_loss(
        self,
        original: Tensor,
        mixed: Tensor,
        disordered: Tensor,
        collective: Tensor,
        beta: Tensor,
    ) -> Tensor:
        """Calculates the Dual-Mixing Contrastive Loss.

        L_user = beta * L_pos + (1 - beta) * L_neg

        L_pos: Anchor=Original. Pos=Mixed. Negs={Disordered, Collective}.
        L_neg: Anchor=Disordered. Pos=Mixed. Negs={Original, Collective}.
        """
        # L_pos: Anchor=Original, Pos=Mixed, HardNegs={Disordered, Collective}
        l_pos = self._hard_nce_loss(
            anchor=original,
            positive=mixed,
            neg_disorder=disordered,
            neg_collective=collective,
            temperature=self.temperature,
        )

        # L_neg: Anchor=Disordered, Pos=Mixed, HardNegs={Original, Collective}
        l_neg = self._hard_nce_loss(
            anchor=disordered,
            positive=mixed,
            neg_disorder=original,
            neg_collective=collective,
            temperature=self.temperature,
        )

        # Weighted Sum
        loss = (beta * l_pos + (1 - beta) * l_neg).mean()
        return loss

    def train_step(self, batch: Any, *args, **kwargs) -> Tensor:
        user, pos_item, neg_item = batch
        batch_size = user.size(0)

        # Get propagated embeddings
        user_all_embeddings, item_all_embeddings = self.forward()

        # Get embeddings for current batch users and items
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # Calculate BPR loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        bpr_pos = self.bpr_loss(pos_scores, neg_scores)

        # Calculate L2 loss
        reg_loss = self.reg_weight * self.reg_loss(
            self.user_embedding(user),
            self.item_embedding(pos_item),
            self.item_embedding(neg_item),
        )

        # Generate Mixing Parameters
        # Beta distribution Beta(alpha, alpha)
        beta_dist = torch.distributions.Beta(self.alpha, self.alpha)
        beta_u = beta_dist.sample((batch_size, 1)).to(self.device)
        beta_i = beta_dist.sample((batch_size, 1)).to(self.device)

        # Disordered Views (Shuffled batch)
        perm_idx = torch.randperm(batch_size, device=self.device)
        u_dis = u_embeddings[perm_idx]
        pos_dis = pos_embeddings[perm_idx]
        neg_dis = neg_embeddings[perm_idx]

        # Individual Mixing
        u_mix = self._mix_embeddings(u_embeddings, u_dis, beta_u)
        pos_mix = self._mix_embeddings(pos_embeddings, pos_dis, beta_i)
        neg_mix = self._mix_embeddings(neg_embeddings, neg_dis, beta_i)

        # Collective Mixing
        u_cm = self._collective_mixing(u_embeddings)
        pos_cm = self._collective_mixing(pos_embeddings)

        # Mixed Negative BPR
        # Encourages item to stay close to user even if mixed with negative
        neg_mix_scores = torch.mul(u_embeddings, neg_mix).sum(dim=1)
        bpr_neg = self.bpr_loss(pos_scores, neg_mix_scores)

        # We use the mean of beta_i for scalar weighting or element-wise
        # Since BPR returns a scalar mean, we use mean of beta
        b_i_scalar = beta_i.mean()
        main_loss = b_i_scalar * bpr_pos + (1 - b_i_scalar) * bpr_neg

        # B. Dual-Mixing Contrastive Loss (Eq. 9 & 10)
        cl_user = self._dual_mixing_cl_loss(u_embeddings, u_mix, u_dis, u_cm, beta_u)
        cl_item = self._dual_mixing_cl_loss(
            pos_embeddings, pos_mix, pos_dis, pos_cm, beta_i
        )

        cl_loss = self.ssl_lambda * (cl_user + cl_item)

        return main_loss + cl_loss + reg_loss

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
        user_all_embeddings, item_all_embeddings = self.propagate_embeddings()

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
