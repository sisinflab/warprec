# pylint: disable = R0801, E1102, R0915
from typing import Any, Optional, List

import torch
from torch import nn, Tensor

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    ContextRecommenderUtils,
)
from warprec.recommenders.layers import MLP
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="NFM")
class NFM(ContextRecommenderUtils, IterativeRecommender):
    """Implementation of NFM algorithm from
        Neural Factorization Machines for Sparse Predictive Analytics, SIGIR 2017.

    For further details, please refer to the `paper <https://arxiv.org/abs/1708.05027>`_.

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
        mlp_hidden_size (List[int]): The MLP hidden layer size list.
        dropout (float): The dropout probability.
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): Number of negative samples for training.
    """

    DATALOADER_TYPE = DataLoaderType.ITEM_RATING_LOADER_WITH_CONTEXT

    embedding_size: int
    mlp_hidden_size: List[int]
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

        # Check for optional value of block size
        self.block_size = kwargs.get("block_size", 50)

        # Ray Tune converts lists to tuples, convert back to list
        self.mlp_hidden_size = list(self.mlp_hidden_size)

        # Batch Normalization after the Bi-Interaction pooling
        self.batch_norm = nn.BatchNorm1d(self.embedding_size)

        # MLP Layers: Input size is the embedding size (output of Bi-Interaction)
        # The MLP class handles the hidden layers and dropout
        self.mlp_layers = MLP(
            [self.embedding_size] + self.mlp_hidden_size, self.dropout
        )

        # Final prediction layer (projects MLP output to scalar)
        self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1, bias=False)

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
        """Forward pass of the NFM model.

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

        # Bi-Interaction Pooling
        sum_of_vectors = torch.sum(fm_input, dim=1)
        sum_of_squares = torch.sum(fm_input.pow(2), dim=1)
        bi_interaction = 0.5 * (sum_of_vectors.pow(2) - sum_of_squares)

        # Neural Layers
        bi_interaction = self.batch_norm(bi_interaction)
        mlp_output = self.mlp_layers(bi_interaction)
        prediction_score = self.predict_layer(mlp_output).squeeze(-1)

        return linear_part + prediction_score

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the NFM model.

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

        # Interaction Fixed Accumulators
        sum_v_fixed = self.user_embedding(user_indices)
        sum_sq_v_fixed = sum_v_fixed.pow(2)

        # Process Contexts (Vettorizzato)
        if contexts is not None and self.context_dims:
            # Linear
            global_ctx = contexts + self.context_offsets
            ctx_bias = self.merged_context_bias(global_ctx).sum(dim=1).squeeze(-1)
            fixed_linear += ctx_bias

            # Interaction
            ctx_emb = self.merged_context_embedding(global_ctx)
            sum_v_fixed += ctx_emb.sum(dim=1)
            sum_sq_v_fixed += ctx_emb.pow(2).sum(dim=1)

        if item_indices is None:
            # Case 'full'
            preds_list = []

            for start in range(0, self.n_items, self.block_size):
                end = min(start + self.block_size, self.n_items)
                current_block_size = end - start

                items_block = torch.arange(start, end, device=user_indices.device)

                # Item Embeddings & Bias
                item_emb = self.item_embedding(items_block)
                item_b = self.item_bias(items_block).squeeze(-1)

                # Feature Embeddings & Bias (Vettorizzato)
                feat_emb_tensor = self._get_feature_embeddings(items_block)
                feat_b = self._get_feature_bias(items_block)

                # Linear Part
                linear_pred = (
                    fixed_linear.unsqueeze(1)
                    + item_b.unsqueeze(0)
                    + feat_b.unsqueeze(0)
                )

                # Bi-Interaction Part
                # Aggregate Item + Features
                if feat_emb_tensor is not None:
                    item_feat_sum = item_emb + feat_emb_tensor.sum(dim=1)
                    item_feat_sq_sum = item_emb.pow(2) + feat_emb_tensor.pow(2).sum(
                        dim=1
                    )
                else:
                    item_feat_sum = item_emb
                    item_feat_sq_sum = item_emb.pow(2)

                # (V_fixed + V_item_total)^2
                sum_v_total = sum_v_fixed.unsqueeze(1) + item_feat_sum.unsqueeze(0)
                sum_v_total_sq = sum_v_total.pow(2)

                # (V_fixed^2 + V_item_total^2)
                sum_sq_total = sum_sq_v_fixed.unsqueeze(1) + item_feat_sq_sum.unsqueeze(
                    0
                )

                # Interaction vector
                bi_interaction = 0.5 * (sum_v_total_sq - sum_sq_total)

                # Flatten for MLP
                bi_interaction_flat = bi_interaction.view(-1, self.embedding_size)

                # Neural Part
                bi_interaction_flat = self.batch_norm(bi_interaction_flat)
                mlp_out = self.mlp_layers(bi_interaction_flat)
                neural_pred = self.predict_layer(mlp_out).view(
                    batch_size, current_block_size
                )

                preds_list.append(linear_pred + neural_pred)

            return torch.cat(preds_list, dim=1)

        # Case 'sampled'
        pad_seq = item_indices.size(1)

        item_emb = self.item_embedding(item_indices)
        item_b = self.item_bias(item_indices).squeeze(-1)

        feat_emb_tensor = self._get_feature_embeddings(item_indices)
        feat_b = self._get_feature_bias(item_indices)

        # Linear Part
        linear_pred = fixed_linear.unsqueeze(1) + item_b + feat_b

        # Bi-Interaction Part
        if feat_emb_tensor is not None:
            item_feat_sum = item_emb + feat_emb_tensor.sum(dim=2)
            item_feat_sq_sum = item_emb.pow(2) + feat_emb_tensor.pow(2).sum(dim=2)
        else:
            item_feat_sum = item_emb
            item_feat_sq_sum = item_emb.pow(2)

        sum_v_fixed_exp = sum_v_fixed.unsqueeze(1)
        sum_sq_v_fixed_exp = sum_sq_v_fixed.unsqueeze(1)

        sum_v_total_sq = (sum_v_fixed_exp + item_feat_sum).pow(2)
        sum_sq_total = sum_sq_v_fixed_exp + item_feat_sq_sum

        bi_interaction = 0.5 * (sum_v_total_sq - sum_sq_total)

        # Flatten for MLP
        bi_interaction_flat = bi_interaction.view(-1, self.embedding_size)

        # Neural Part
        bi_interaction_flat = self.batch_norm(bi_interaction_flat)
        mlp_out = self.mlp_layers(bi_interaction_flat)
        neural_pred = self.predict_layer(mlp_out).view(batch_size, pad_seq)

        return linear_pred + neural_pred
