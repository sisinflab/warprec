# pylint: disable = R0801, E1102
from typing import Any, Optional, Tuple

import torch
from torch import nn, Tensor

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    ContextRecommenderUtils,
)
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


class AttentionLayer(nn.Module):
    """Implements the Attention Network.

    Equation: a_ij = h^T ReLU(W(v_i * v_j) + b)

    Args:
        embedding_size (int): The embedding size value.
        attention_size (int): The attention size value.
    """

    def __init__(self, embedding_size: int, attention_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, attention_size),
            nn.ReLU(),
            nn.Linear(attention_size, 1),
        )

    def logits(self, x: Tensor) -> Tensor:
        """The unnormalised attention score of each pair.

        Prediction normalises over pair groups that are computed separately, so
        it needs the logits before the softmax rather than after it.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The unnormalised score tensor.
        """
        return self.mlp(x)

    def forward(self, x: Tensor) -> Tensor:
        """The forward step of the attention layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The score tensor.
        """
        # x: [batch_size, num_pairs, embedding_size]
        # scores: [batch_size, num_pairs, 1]
        return torch.softmax(self.logits(x), dim=1)


@model_registry.register(name="AFM")
class AFM(ContextRecommenderUtils, IterativeRecommender):
    """Implementation of AFM algorithm from
        Attentional Factorization Machines: Learning the Weight of Feature Interactions
        via Attention Networks, IJCAI 2017.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        interactions (Optional[Interactions]): The training interactions.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The size of the latent vectors.
        attention_size (int): The size of the attention network hidden layer.
        dropout (float): The dropout probability.
        reg_weight (float): The L2 regularization weight for embeddings.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): Number of negative samples for training.
    """

    DATALOADER_TYPE = DataLoaderType.ITEM_RATING_LOADER_WITH_CONTEXT

    embedding_size: int
    attention_size: int
    dropout: float
    reg_weight: float
    weight_decay: float
    batch_size: int
    epochs: int
    learning_rate: float
    neg_samples: int

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        interactions: Optional[Interactions] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            params, info, *args, interactions=interactions, seed=seed, **kwargs
        )

        self.block_size = kwargs.get("block_size", 50)
        self.chunk_size = kwargs.get("chunk_size", 4096)

        # Attention Network
        self.attention_layer = AttentionLayer(self.embedding_size, self.attention_size)

        # Projection Vector p
        # Weights the final pooled vector to produce the score
        self.p = nn.Parameter(torch.randn(self.embedding_size))

        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        # Pair indices of a field group, built on first use
        self._pair_cache: dict[int, Tensor] = {}

        # Pre-compute Pair Indices
        # Total fields = User (1) + Item (1) + Features (N) + Contexts (M)
        self.num_fields = 2 + len(self.feature_labels) + len(self.context_labels)

        # Generate indices for all unique pairs (i, j) where i < j
        row_idx = []
        col_idx = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                row_idx.append(i)
                col_idx.append(j)

        # Register as buffers
        self.register_buffer("p_idx", torch.tensor(row_idx, dtype=torch.long))
        self.register_buffer("q_idx", torch.tensor(col_idx, dtype=torch.long))

        # Losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = EmbLoss()

        self.apply(self._init_weights)

    def _compute_afm_interaction(self, stacked_embeddings: Tensor) -> Tensor:
        """Computes the AFM interaction part."""
        # Pair-wise Interaction Layer
        # [batch_size, num_pairs, embedding_size]
        p = stacked_embeddings[:, self.p_idx]  # type: ignore[index]
        q = stacked_embeddings[:, self.q_idx]  # type: ignore[index]

        # Element-wise product
        pair_wise_inter = p * q

        # Apply Dropout on the interaction vectors
        pair_wise_inter = self.dropout_layer(pair_wise_inter)

        # Attention-based Pooling
        att_weights = self.attention_layer(
            pair_wise_inter
        )  # [batch_size, num_pairs, 1]

        # Weighted sum
        att_pooling = torch.sum(
            att_weights * pair_wise_inter, dim=1
        )  # [batch_size, embedding_size]

        # Final Projection
        afm_score = torch.sum(att_pooling * self.p, dim=1)  # [batch_size]

        return afm_score

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        user, item, rating = batch[0], batch[1], batch[2]

        contexts: Optional[Tensor] = None
        features: Optional[Tensor] = None

        current_idx = 3

        # If feature dimensions exist, the next element is features
        if self.feature_dims:
            features = batch[current_idx]
            current_idx += 1

        # If context dimensions exist, the next element is context
        if self.context_dims:
            contexts = batch[current_idx]

        prediction = self.forward(user, item, features, contexts)

        # Compute BCE loss
        bce_loss = self.bce_loss(prediction, rating)

        # Compute L2 regularization on embeddings and biases
        reg_params = self.get_reg_params(user, item, features, contexts)
        reg_loss = self.reg_weight * self.reg_loss(*reg_params)

        # Loss logging
        loss = bce_loss + reg_loss
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def forward(
        self,
        user: Tensor,
        item: Tensor,
        features: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
    ) -> Tensor:
        # Linear Part (First Order)
        linear_part = self.compute_first_order(user, item, features, contexts)

        # Interaction Part (Second Order)
        u_emb = self.user_embedding(user).unsqueeze(1)
        i_emb = self.item_embedding(item).unsqueeze(1)
        components = [u_emb, i_emb]

        # Add Feature Embeddings
        if features is not None and self.feature_dims:
            global_feat = features + self.feature_offsets
            f_emb = self.merged_feature_embedding(global_feat)
            components.append(f_emb)

        # Add Context Embeddings
        if contexts is not None and self.context_labels:
            c_emb = self._get_context_embeddings(contexts)
            components.append(c_emb)

        # Concatenate on Field dimension
        stacked_embeddings = torch.cat(components, dim=1)

        # AFM Interaction Part
        afm_part = self._compute_afm_interaction(stacked_embeddings)

        return linear_part + afm_part

    def _pair_indices(self, num_fields: int, device: torch.device) -> Tensor:
        """The (i, j) index pairs of one field group, cached per group size.

        Args:
            num_fields (int): The number of fields in the group.
            device (torch.device): The device the indices are needed on.

        Returns:
            Tensor: The [2, num_pairs] index tensor, empty when the group holds
                fewer than two fields.
        """
        cached = self._pair_cache.get(num_fields)
        if cached is None or cached.device != device:
            cached = torch.triu_indices(num_fields, num_fields, offset=1, device=device)
            self._pair_cache[num_fields] = cached
        return cached

    def _group_terms(self, pairs: Tensor) -> Tuple[Tensor, Tensor]:
        """Reduce a group of pair vectors to its attention logit and its score.

        The projection onto ``p`` is linear, so it can be applied to each pair
        before the attention weights rather than to their weighted sum. That
        drops the embedding dimension immediately and leaves one scalar per
        pair to carry around.

        Args:
            pairs (Tensor): The pair vectors, [..., num_pairs, embedding_size].

        Returns:
            Tuple[Tensor, Tensor]: The logits and the projected scores, both
                [..., num_pairs].
        """
        pairs = self.dropout_layer(pairs)
        return (
            self.attention_layer.logits(pairs).squeeze(-1),
            (pairs * self.p).sum(-1),
        )

    def _compute_network_scores(
        self,
        u_emb: Tensor,
        item_emb: Tensor,
        feat_emb_tensor: Optional[Tensor],
        ctx_emb_tensor: Optional[Tensor],
        batch_size: int,
        num_items: int,
    ) -> Tensor:
        """Compute the AFM interaction part for a block of candidate items.

        The fields split in two: the user and the contexts are fixed for a row,
        while the item and its features change with the candidate. A pair drawn
        from the fixed side is therefore identical for every candidate, and a
        pair drawn from the varying side is identical for every row, so only the
        pairs that cross the two sides depend on both. Computing the other two
        groups once is an exact rewrite, not an approximation: the softmax still
        normalises over the same set of pairs, and a sum does not care in which
        order they arrive.

        Args:
            u_emb (Tensor): The user embeddings, [batch_size, embedding_size].
            item_emb (Tensor): The candidate item embeddings,
                [num_items, embedding_size].
            feat_emb_tensor (Optional[Tensor]): The item feature embeddings,
                [num_items, num_features, embedding_size].
            ctx_emb_tensor (Optional[Tensor]): The context embeddings,
                [batch_size, num_contexts, embedding_size].
            batch_size (int): The number of rows scored.
            num_items (int): The number of candidate items in the block.

        Returns:
            Tensor: The interaction scores, [batch_size, num_items].
        """
        device = u_emb.device

        # Fields that are fixed for a row, and fields that follow the candidate
        fixed = [u_emb.unsqueeze(1)]
        if ctx_emb_tensor is not None:
            fixed.append(ctx_emb_tensor)
        fixed_emb = torch.cat(fixed, dim=1)  # [batch_size, n_fixed, emb]

        varying = [item_emb.unsqueeze(1)]
        if feat_emb_tensor is not None:
            varying.append(feat_emb_tensor)
        var_emb = torch.cat(varying, dim=1)  # [num_items, n_varying, emb]

        n_fixed = fixed_emb.size(1)
        n_varying = var_emb.size(1)

        logits: list[Tensor] = []
        scores: list[Tensor] = []

        # Fixed x fixed: one value per row, shared by every candidate
        if n_fixed > 1:
            idx = self._pair_indices(n_fixed, device)
            ff_logit, ff_score = self._group_terms(
                fixed_emb[:, idx[0]] * fixed_emb[:, idx[1]]
            )
            logits.append(ff_logit.unsqueeze(1).expand(-1, num_items, -1))
            scores.append(ff_score.unsqueeze(1).expand(-1, num_items, -1))

        # Varying x varying: one value per candidate, shared by every row
        if n_varying > 1:
            idx = self._pair_indices(n_varying, device)
            vv_logit, vv_score = self._group_terms(
                var_emb[:, idx[0]] * var_emb[:, idx[1]]
            )
            logits.append(vv_logit.unsqueeze(0).expand(batch_size, -1, -1))
            scores.append(vv_score.unsqueeze(0).expand(batch_size, -1, -1))

        # Fixed x varying: the only group that genuinely depends on both, and
        # the only one still paid for per (row, candidate)
        cross = fixed_emb.unsqueeze(1).unsqueeze(3) * var_emb.unsqueeze(0).unsqueeze(2)
        cross = cross.reshape(batch_size, num_items, n_fixed * n_varying, -1)
        fv_logit, fv_score = self._group_terms(cross)
        logits.append(fv_logit)
        scores.append(fv_score)

        # One softmax over every pair, exactly as the monolithic path does
        weights = torch.softmax(torch.cat(logits, dim=2), dim=2)
        return (weights * torch.cat(scores, dim=2)).sum(-1)

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the AFM model.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            contexts (Optional[Tensor]): The batch of contexts.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        batch_size = user_indices.size(0)

        # Linear Fixed
        fixed_linear = self.global_bias + self.user_bias(user_indices).squeeze(-1)
        if contexts is not None and self.context_dims:
            ctx_bias = self._get_context_bias(contexts)
            fixed_linear += ctx_bias

        # Embeddings Fixed
        u_emb = self.user_embedding(user_indices)  # [batch_size, embedding_size]
        ctx_emb_tensor = self._get_context_embeddings(contexts)

        if item_indices is None:
            # Case 'full': iterate through all items in memory-safe blocks
            preds_list = []

            # The catalogue does not change while the users are scored, so
            # the item-side tensors are gathered once and sliced per block.
            (
                cat_item_emb,
                cat_item_bias,
                cat_feat_emb,
                cat_feat_bias,
            ) = self._catalogue_item_side()

            for start in range(0, self.n_items, self.block_size):
                end = min(start + self.block_size, self.n_items)
                current_block_len = end - start

                # Slices of the catalogue tensors, shared by every user
                item_emb_block = cat_item_emb[start:end]
                item_bias_block = cat_item_bias[start:end]
                feat_emb_block_tensor = (
                    None if cat_feat_emb is None else cat_feat_emb[start:end]
                )
                feat_bias_block = cat_feat_bias[start:end]

                # Linear Part
                linear_pred = (
                    fixed_linear.unsqueeze(1)
                    + item_bias_block.unsqueeze(0)
                    + feat_bias_block.unsqueeze(0)
                )

                # Compute AFM scores efficiently
                afm_scores = self._compute_network_scores(
                    u_emb,
                    item_emb_block,
                    feat_emb_block_tensor,
                    ctx_emb_tensor,
                    batch_size,
                    current_block_len,
                )

                preds_list.append(linear_pred + afm_scores)

            return torch.cat(preds_list, dim=1)

        # Case 'sampled': process given item_indices
        pad_seq = item_indices.size(1)

        # Item Embeddings: [Batch, Seq, Emb]
        item_emb = self.item_embedding(item_indices)

        # Retrieve item feature embeddings & bias
        # feat_emb_tensor: [Batch, Seq, Num_Feat, Emb]
        feat_emb_tensor = self._get_feature_embeddings(item_indices)
        feat_bias = self._get_feature_bias(item_indices)

        # Linear
        item_bias = self.item_bias(item_indices).squeeze(-1)
        linear_pred = fixed_linear.unsqueeze(1) + item_bias + feat_bias

        # Stack Construction
        # User: [Batch, 1, 1, Emb] -> [Batch, Seq, 1, Emb]
        u_emb_exp = u_emb.unsqueeze(1).unsqueeze(2).expand(-1, pad_seq, -1, -1)

        # Item: [Batch, Seq, Emb] -> [Batch, Seq, 1, Emb]
        i_emb_exp = item_emb.unsqueeze(2)

        stack_list = [u_emb_exp, i_emb_exp]

        if feat_emb_tensor is not None:
            stack_list.append(feat_emb_tensor)

        if ctx_emb_tensor is not None:
            # Context: [Batch, Num_Ctx, Emb] -> [Batch, 1, Num_Ctx, Emb] -> [Batch, Seq, Num_Ctx, Emb]
            c_emb_exp = ctx_emb_tensor.unsqueeze(1).expand(-1, pad_seq, -1, -1)
            stack_list.append(c_emb_exp)

        # Concatenate on Field dimension (dim=2)
        # [Batch, Seq, Total_Fields, Emb]
        stack = torch.cat(stack_list, dim=2)

        # Reshape to [Batch * Seq, Total_Fields, Emb]
        total_rows = batch_size * pad_seq
        stack_flat = stack.view(total_rows, self.num_fields, self.embedding_size)

        # AFM part
        afm_scores_flat = self._compute_afm_interaction(stack_flat)
        afm_scores = afm_scores_flat.view(batch_size, pad_seq)

        return linear_pred + afm_scores
