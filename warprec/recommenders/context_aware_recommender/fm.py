import torch
from torch import nn, Tensor
from typing import Any, Optional

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    ContextRecommenderUtils,
)
from warprec.recommenders.layers import FactorizationMachine
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="FM")
class FM(ContextRecommenderUtils, IterativeRecommender):
    """Implementation of FM algorithm from
        Factorization Machines ICDM 2010.

    For further details, check the `paper <https://ieeexplore.ieee.org/document/5694074>`_.

    Args:
        params (dict): Model parameters.
        interactions (Interactions): The training interactions.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The size of the latent vectors.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): Number of negative samples for training.
    """

    DATALOADER_TYPE = DataLoaderType.ITEM_RATING_LOADER_WITH_CONTEXT

    embedding_size: int
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float
    neg_samples: int

    def __init__(
        self,
        params: dict,
        interactions: Interactions,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, info, *args, seed=seed, **kwargs)

        # FM Layer (Interaction Part - Second Order)
        self.fm = FactorizationMachine(reduce_sum=True)

        # Losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = EmbLoss()

        # Initialize weights
        self.apply(self._init_weights)

    def train_step(self, batch: Any, epoch: int, *args, **kwargs) -> Tensor:
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
        loss = self.bce_loss(prediction, rating)

        # Compute L2 regularization on embeddings and biases
        reg_params = self.get_reg_params(user, item, features, contexts)
        reg_loss = self.reg_weight * self.reg_loss(*reg_params)

        return loss + reg_loss

    def forward(
        self,
        user: Tensor,
        item: Tensor,
        features: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the FM model.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.
            features (Optional[Tensor]): The tensor containing the features of the interactions.
            contexts (Optional[Tensor]): The tensor containing the context of the interactions.

        Returns:
            Tensor: The prediction score for each triplet (user, item, context).
        """
        # Linear Part
        linear_part = self.compute_first_order(user, item, features, contexts)

        # Interaction Part (Second Order)
        embeddings_list = [self.user_embedding(user), self.item_embedding(item)]

        # Add Feature Embeddings
        if features is not None and self.feature_labels:
            for idx, name in enumerate(self.feature_labels):
                embeddings_list.append(self.feature_embedding[name](features[:, idx]))

        # Add Context Embeddings
        if contexts is not None and self.context_labels:
            for idx, name in enumerate(self.context_labels):
                embeddings_list.append(self.context_embedding[name](contexts[:, idx]))

        fm_input = torch.stack(embeddings_list, dim=1)
        interaction_part = self.fm(fm_input).squeeze(-1)

        return linear_part + interaction_part

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the linear part and FM.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            contexts (Optional[Tensor]): The batch of contexts. Required to
                predict with CARS models.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        # Linear Fixed
        fixed_linear = self.global_bias + self.user_bias(user_indices).squeeze(-1)

        # FM Fixed Accumulators (Sum V and Sum V^2)
        sum_v_fixed = self.user_embedding(user_indices)
        sum_sq_v_fixed = sum_v_fixed.pow(2)

        # Process Contexts
        if contexts is not None and self.context_labels:
            for idx, name in enumerate(self.context_labels):
                ctx_input = contexts[:, idx]

                # Linear
                fixed_linear += self.context_bias[name](ctx_input).squeeze(-1)

                # Interaction
                ctx_emb = self.context_embedding[name](ctx_input)
                sum_v_fixed += ctx_emb
                sum_sq_v_fixed += ctx_emb.pow(2)

        # Determine target items
        if item_indices is None:
            # All items (excluding padding)
            target_items = torch.arange(self.n_items, device=fixed_linear.device)
        else:
            target_items = item_indices
            if target_items.dim() == 1:
                target_items = target_items.unsqueeze(
                    1
                )  # [batch_size, 1] for sampled case

        # Calculate Total Item Linear Bias (Bias_i + Sum Bias_features)
        item_linear_total = self.item_bias(target_items).squeeze(-1)

        # Calculate Total Item Embedding (V_i + Sum V_features) -> For (Sum V)^2 term
        item_emb_sum = self.item_embedding(target_items)

        # Calculate Total Item Squared Embedding (V_i^2 + Sum V_features^2) -> For Sum (V^2) term
        item_emb_sq_sum = self.item_embedding(target_items).pow(2)

        # Add Features if available
        if self.feature_dims and self.item_features is not None:
            # Flatten target_items for indexing, then reshape back
            flat_items = target_items.view(-1).cpu()
            item_feats = self.item_features[flat_items].to(fixed_linear.device)

            # Reshape to [n_items, n_features] or [batch_size, 1, n_features]
            target_shape = target_items.shape

            for idx, name in enumerate(self.feature_labels):
                feat_col = item_feats[:, idx].view(target_shape)

                # Linear Bias
                feat_bias = self.feature_bias[name](feat_col).squeeze(-1)
                item_linear_total += feat_bias

                # Embeddings
                feat_emb = self.feature_embedding[name](feat_col)

                # Update Sum (for the first part of FM equation)
                item_emb_sum += feat_emb

                # Update Sum Sq (for the second part of FM equation)
                item_emb_sq_sum += feat_emb.pow(2)

        if item_indices is None:
            # Case 'full': [batch_size, n_items]
            # Broadcast Fixed (Batch, 1) vs Variable (1, Items)

            final_linear = fixed_linear.unsqueeze(1) + item_linear_total.unsqueeze(0)

            # Prepare for broadcasting
            sum_v_fixed_exp = sum_v_fixed.unsqueeze(1)  # [batch_size, 1, emb_size]
            sum_sq_v_fixed_exp = sum_sq_v_fixed.unsqueeze(
                1
            )  # [batch_size, 1, emb_size]

            item_emb_sum_exp = item_emb_sum.unsqueeze(0)  # [1, n_items, emb_size]
            item_emb_sq_sum_exp = item_emb_sq_sum.unsqueeze(0)  # [1, n_items, emb_size]

            # FM Equation: 0.5 * ( (Sum V)^2 - Sum (V^2) )

            # (Sum V)^2 = (V_fixed + V_item_total)^2
            sum_all_sq = (sum_v_fixed_exp + item_emb_sum_exp).pow(2)

            # Sum (V^2) = V_fixed^2 + V_item_total_sq
            sum_sq_all = sum_sq_v_fixed_exp + item_emb_sq_sum_exp

            interaction = 0.5 * (sum_all_sq - sum_sq_all).sum(dim=2)

        else:
            # Case 'sampled': [batch_size, 1]
            # Dimensions match directly or broadcast on batch

            final_linear = fixed_linear.unsqueeze(1) + item_linear_total

            sum_v_fixed_exp = sum_v_fixed.unsqueeze(1)
            sum_sq_v_fixed_exp = sum_sq_v_fixed.unsqueeze(1)

            # (Sum V)^2
            sum_all_sq = (sum_v_fixed_exp + item_emb_sum).pow(2)

            # Sum (V^2)
            sum_sq_all = sum_sq_v_fixed_exp + item_emb_sq_sum

            interaction = 0.5 * (sum_all_sq - sum_sq_all).sum(dim=2)

        return final_linear + interaction
