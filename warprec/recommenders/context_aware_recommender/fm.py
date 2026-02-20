# pylint: disable = R0801, E1102
from typing import Any, Optional

import torch
from torch import nn, Tensor

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

    For further details, please refer to the `paper <https://ieeexplore.ieee.org/document/5694074>`_.

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
        info: dict,
        *args: Any,
        interactions: Optional[Interactions] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            params, info, *args, interactions=interactions, seed=seed, **kwargs
        )

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
            global_ctx = contexts + self.context_offsets
            c_emb = self.merged_context_embedding(global_ctx)
            components.append(c_emb)

        fm_input = torch.cat(components, dim=1)
        interaction_part = self.fm(fm_input).squeeze(-1)

        return linear_part + interaction_part

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
            # Linear Context
            global_ctx = contexts + self.context_offsets
            ctx_bias = self.merged_context_bias(global_ctx).sum(dim=1).squeeze(-1)
            fixed_linear += ctx_bias

            # FM Context
            ctx_emb = self.merged_context_embedding(global_ctx)
            sum_v_fixed += ctx_emb.sum(dim=1)
            sum_sq_v_fixed += ctx_emb.pow(2).sum(dim=1)

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

        # Item Linear Bias
        item_linear_total = self.item_bias(target_items).squeeze(-1)

        # Item Embeddings
        item_emb = self.item_embedding(target_items)

        # Feature Handling
        feat_bias = self._get_feature_bias(target_items)
        feat_emb_tensor = self._get_feature_embeddings(target_items)

        # Update Linear
        item_linear_total += feat_bias

        # Update FM accumulators
        if feat_emb_tensor is not None:
            feat_sum = feat_emb_tensor.sum(dim=-2)
            feat_sq_sum = feat_emb_tensor.pow(2).sum(dim=-2)

            # Total Item Component
            item_component_sum = item_emb + feat_sum
            item_component_sq_sum = item_emb.pow(2) + feat_sq_sum
        else:
            item_component_sum = item_emb
            item_component_sq_sum = item_emb.pow(2)

        if item_indices is None:
            # Case 'full': [batch_size, n_items]

            final_linear = fixed_linear.unsqueeze(1) + item_linear_total.unsqueeze(0)

            # Prepare for broadcasting
            sum_v_fixed_exp = sum_v_fixed.unsqueeze(1)  # [B, 1, E]
            sum_sq_v_fixed_exp = sum_sq_v_fixed.unsqueeze(1)  # [B, 1, E]

            item_sum_exp = item_component_sum.unsqueeze(0)  # [1, I, E]
            item_sq_sum_exp = item_component_sq_sum.unsqueeze(0)  # [1, I, E]

            # FM Equation
            sum_all_sq = (sum_v_fixed_exp + item_sum_exp).pow(2)
            sum_sq_all = sum_sq_v_fixed_exp + item_sq_sum_exp

            interaction = 0.5 * (sum_all_sq - sum_sq_all).sum(dim=2)
        else:
            # Case 'sampled': [batch_size, 1]

            final_linear = fixed_linear.unsqueeze(1) + item_linear_total

            sum_v_fixed_exp = sum_v_fixed.unsqueeze(1)
            sum_sq_v_fixed_exp = sum_sq_v_fixed.unsqueeze(1)

            sum_all_sq = (sum_v_fixed_exp + item_component_sum).pow(2)
            sum_sq_all = sum_sq_v_fixed_exp + item_component_sq_sum

            interaction = 0.5 * (sum_all_sq - sum_sq_all).sum(dim=2)

        return final_linear + interaction
