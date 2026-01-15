# pylint: disable = R0801, E1102, R0915
from typing import Any, Optional, List

import torch
from torch import nn, Tensor

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    ContextRecommenderUtils,
)
from warprec.recommenders.layers import FactorizationMachine, MLP
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="DeepFM")
class DeepFM(ContextRecommenderUtils, IterativeRecommender):
    """Implementation of DeepFM algorithm from
        DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, IJCAI 2017.

    For further details, check the `paper <https://arxiv.org/abs/1703.04247>`_.

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
        interactions: Interactions,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, info, *args, seed=seed, **kwargs)

        # Check for optional value of block size
        self.block_size = kwargs.get("block_size", 50)

        # Ray Tune converts lists to tuples, convert back to list
        self.mlp_hidden_size = list(self.mlp_hidden_size)

        # FM Layer (Interaction Part - Second Order)
        self.fm = FactorizationMachine(reduce_sum=True)

        # Deep Part (DNN)
        # Calculate total number of fields: User + Item + Features + Contexts
        self.num_fields = 2 + len(self.feature_labels) + len(self.context_labels)

        # Input size for MLP is the concatenation of all embeddings
        input_dim = self.num_fields * self.embedding_size

        self.mlp_layers = MLP([input_dim] + self.mlp_hidden_size, self.dropout)

        # Final prediction layer for the Deep part
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

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
        """Forward pass of the DeepFM model.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.
            features (Optional[Tensor]): The tensor containing the features of the interactions.
            contexts (Optional[Tensor]): The tensor containing the context of the interactions.

        Returns:
            Tensor: The prediction score.
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

        stacked_embeddings = torch.cat(components, dim=1)

        # FM Interaction
        fm_part = self.fm(stacked_embeddings).squeeze(-1)

        # Deep Interaction
        batch_size = stacked_embeddings.shape[0]
        deep_input = stacked_embeddings.view(batch_size, -1)
        deep_output = self.mlp_layers(deep_input)
        deep_part = self.deep_predict_layer(deep_output).squeeze(-1)

        return linear_part + fm_part + deep_part

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the DeepFM model.

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

        # FM Fixed Accumulators
        sum_v_fixed = self.user_embedding(user_indices)
        sum_sq_v_fixed = sum_v_fixed.pow(2)

        # Deep Fixed Parts
        user_emb = self.user_embedding(user_indices)

        # Contexts
        ctx_emb_tensor = self._get_context_embeddings(contexts)

        # Process Contexts
        if contexts is not None and self.context_dims:
            # Linear
            global_ctx = contexts + self.context_offsets
            ctx_bias = self.merged_context_bias(global_ctx).sum(dim=1).squeeze(-1)
            fixed_linear += ctx_bias

            # FM
            sum_v_fixed += ctx_emb_tensor.sum(dim=1)
            sum_sq_v_fixed += ctx_emb_tensor.pow(2).sum(dim=1)

        if item_indices is None:
            # Case 'full'
            preds_list = []

            for start in range(0, self.n_items, self.block_size):
                end = min(start + self.block_size, self.n_items)
                current_block_size = end - start

                items_block = torch.arange(start, end, device=user_indices.device)

                # Item Embeddings and Bias
                item_emb = self.item_embedding(items_block)
                item_b = self.item_bias(items_block).squeeze(-1)

                # Feature Embeddings and Bias
                # feat_emb_tensor: [Block, Num_Feat, Emb]
                feat_emb_tensor = self._get_feature_embeddings(items_block)
                feat_b = self._get_feature_bias(items_block)

                # Linear Part
                linear_pred = (
                    fixed_linear.unsqueeze(1)
                    + item_b.unsqueeze(0)
                    + feat_b.unsqueeze(0)
                )

                # FM Part
                # Aggregate Item + Features
                if feat_emb_tensor is not None:
                    item_feat_sum = item_emb + feat_emb_tensor.sum(dim=1)
                    item_feat_sq_sum = item_emb.pow(2) + feat_emb_tensor.pow(2).sum(
                        dim=1
                    )
                else:
                    item_feat_sum = item_emb
                    item_feat_sq_sum = item_emb.pow(2)

                sum_v_total = sum_v_fixed.unsqueeze(1) + item_feat_sum.unsqueeze(0)
                sum_v_total_sq = sum_v_total.pow(2)

                sum_sq_total = sum_sq_v_fixed.unsqueeze(1) + item_feat_sq_sum.unsqueeze(
                    0
                )

                fm_pred = 0.5 * (sum_v_total_sq - sum_sq_total).sum(dim=2)

                # Deep Part
                # Expand User: [Batch, 1, 1, Emb] -> [Batch, Block, 1, Emb]
                u_exp = (
                    user_emb.unsqueeze(1)
                    .unsqueeze(2)
                    .expand(-1, current_block_size, -1, -1)
                )

                # Expand Item: [1, Block, 1, Emb] -> [Batch, Block, 1, Emb]
                i_exp = (
                    item_emb.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1, -1)
                )

                stack_list = [u_exp, i_exp]

                # Expand Features: [1, Block, N_Feat, Emb] -> [Batch, Block, N_Feat, Emb]
                if feat_emb_tensor is not None:
                    f_exp = feat_emb_tensor.unsqueeze(0).expand(batch_size, -1, -1, -1)
                    stack_list.append(f_exp)

                # Expand Contexts: [Batch, 1, N_Ctx, Emb] -> [Batch, Block, N_Ctx, Emb]
                if ctx_emb_tensor is not None:
                    c_exp = ctx_emb_tensor.unsqueeze(1).expand(
                        -1, current_block_size, -1, -1
                    )
                    stack_list.append(c_exp)

                # Concatenate: [Batch, Block, Total_Fields, Emb]
                deep_input_block = torch.cat(stack_list, dim=2)

                deep_input_flat = deep_input_block.view(
                    -1, self.num_fields * self.embedding_size
                )

                deep_out = self.mlp_layers(deep_input_flat)
                deep_pred = self.deep_predict_layer(deep_out).view(
                    batch_size, current_block_size
                )

                preds_list.append(linear_pred + fm_pred + deep_pred)

            return torch.cat(preds_list, dim=1)
        # Case 'sampled'
        pad_seq = item_indices.size(1)

        item_emb = self.item_embedding(item_indices)
        item_b = self.item_bias(item_indices).squeeze(-1)

        feat_emb_tensor = self._get_feature_embeddings(item_indices)
        feat_b = self._get_feature_bias(item_indices)

        # Linear Part
        linear_pred = fixed_linear.unsqueeze(1) + item_b + feat_b

        # FM Part
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

        fm_pred = 0.5 * (sum_v_total_sq - sum_sq_total).sum(dim=2)

        # Deep Part
        # User: [Batch, Seq, 1, Emb]
        u_exp = user_emb.unsqueeze(1).unsqueeze(2).expand(-1, pad_seq, -1, -1)

        # Item: [Batch, Seq, 1, Emb]
        i_exp = item_emb.unsqueeze(2)

        stack_list = [u_exp, i_exp]

        # Features: [Batch, Seq, N_Feat, Emb] (Already correct)
        if feat_emb_tensor is not None:
            stack_list.append(feat_emb_tensor)

        # Contexts: [Batch, Seq, N_Ctx, Emb]
        if ctx_emb_tensor is not None:
            c_exp = ctx_emb_tensor.unsqueeze(1).expand(-1, pad_seq, -1, -1)
            stack_list.append(c_exp)

        deep_input_block = torch.cat(stack_list, dim=2)
        deep_input_flat = deep_input_block.view(
            -1, self.num_fields * self.embedding_size
        )

        deep_out = self.mlp_layers(deep_input_flat)
        deep_pred = self.deep_predict_layer(deep_out).view(batch_size, pad_seq)

        return linear_pred + fm_pred + deep_pred
