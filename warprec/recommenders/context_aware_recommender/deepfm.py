import torch
from torch import nn, Tensor
from typing import Any, Optional, List

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

        # Prepare Embeddings
        embeddings_list = [self.user_embedding(user), self.item_embedding(item)]

        # Add Feature Embeddings
        if features is not None and self.feature_labels:
            for idx, name in enumerate(self.feature_labels):
                embeddings_list.append(self.feature_embedding[name](features[:, idx]))

        # Add Context Embeddings
        if contexts is not None and self.context_labels:
            for idx, name in enumerate(self.context_labels):
                embeddings_list.append(self.context_embedding[name](contexts[:, idx]))

        stacked_embeddings = torch.stack(embeddings_list, dim=1)

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

        # Deep Fixed Parts (List of embeddings to be concatenated)
        user_emb = self.user_embedding(user_indices)
        ctx_emb_list = []

        # Process Contexts
        if contexts is not None and self.context_labels:
            for idx, name in enumerate(self.context_labels):
                ctx_input = contexts[:, idx]

                # Linear
                fixed_linear += self.context_bias[name](ctx_input).squeeze(-1)

                # FM Interaction
                ctx_emb = self.context_embedding[name](ctx_input)
                sum_v_fixed += ctx_emb
                sum_sq_v_fixed += ctx_emb.pow(2)

                # Deep
                ctx_emb_list.append(ctx_emb)

        # Helper to retrieve feature embeddings for a set of items
        def get_feature_embeddings(target_items: Tensor) -> List[Tensor]:
            feat_embs = []
            if self.feature_dims and self.item_features is not None:
                flat_items = target_items.view(-1).cpu()
                item_feats = self.item_features[flat_items].to(self.device)
                target_shape = target_items.shape

                for idx, name in enumerate(self.feature_labels):
                    feat_col = item_feats[:, idx].view(target_shape)
                    feat_embs.append(self.feature_embedding[name](feat_col))
            return feat_embs

        # Helper to retrieve feature biases
        def get_feature_biases(target_items: Tensor) -> Tensor:
            feat_bias_sum = torch.zeros(target_items.shape, device=self.device)
            if self.feature_dims and self.item_features is not None:
                flat_items = target_items.view(-1).cpu()
                item_feats = self.item_features[flat_items].to(self.device)
                target_shape = target_items.shape

                for idx, name in enumerate(self.feature_labels):
                    feat_col = item_feats[:, idx].view(target_shape)
                    feat_bias_sum += self.feature_bias[name](feat_col).squeeze(-1)
            return feat_bias_sum

        if item_indices is None:
            # Case 'full': iterate through all items in memory-safe blocks
            preds_list = []

            for start in range(0, self.n_items, self.block_size):
                end = min(start + self.block_size, self.n_items)
                current_block_size = end - start

                # Item indices for this block
                items_block = torch.arange(start, end, device=user_indices.device)

                # Get Item Embeddings and Bias
                item_emb = self.item_embedding(
                    items_block
                )  # [block_size, embedding_size]
                item_b = self.item_bias(items_block).squeeze(-1)  # [block_size]

                # Get Feature Embeddings and Bias
                feat_emb_list = get_feature_embeddings(
                    items_block
                )  # List of [block_size, emb]
                feat_b = get_feature_biases(items_block)  # [block_size]

                # Linear Part
                linear_pred = (
                    fixed_linear.unsqueeze(1)
                    + item_b.unsqueeze(0)
                    + feat_b.unsqueeze(0)
                )

                # FM Part
                # Aggregate Item + Features for FM (Vector Sum Trick)
                item_feat_sum = item_emb
                item_feat_sq_sum = item_emb.pow(2)

                for f_emb in feat_emb_list:
                    item_feat_sum = item_feat_sum + f_emb
                    item_feat_sq_sum = item_feat_sq_sum + f_emb.pow(2)

                # (V_fixed + V_item_total)^2
                sum_v_total = sum_v_fixed.unsqueeze(1) + item_feat_sum.unsqueeze(0)
                sum_v_total_sq = sum_v_total.pow(2)

                # (V_fixed^2 + V_item_total^2)
                sum_sq_total = sum_sq_v_fixed.unsqueeze(1) + item_feat_sq_sum.unsqueeze(
                    0
                )

                fm_pred = 0.5 * (sum_v_total_sq - sum_sq_total).sum(dim=2)

                # Deep Part
                # Expand User & Contexts
                user_emb_exp = user_emb.unsqueeze(1).expand(-1, current_block_size, -1)
                ctx_emb_exp_list = [
                    c.unsqueeze(1).expand(-1, current_block_size, -1)
                    for c in ctx_emb_list
                ]

                # Expand Item & Features
                item_emb_exp = item_emb.unsqueeze(0).expand(batch_size, -1, -1)
                feat_emb_exp_list = [
                    f.unsqueeze(0).expand(batch_size, -1, -1) for f in feat_emb_list
                ]

                # Concatenate: User, Item, Features, Contexts
                deep_input_block = torch.cat(
                    [user_emb_exp, item_emb_exp] + feat_emb_exp_list + ctx_emb_exp_list,
                    dim=2,
                )

                deep_input_flat = deep_input_block.view(
                    -1, self.num_fields * self.embedding_size
                )

                deep_out = self.mlp_layers(deep_input_flat)
                deep_pred = self.deep_predict_layer(deep_out).view(
                    batch_size, current_block_size
                )

                # Sum and append
                preds_list.append(linear_pred + fm_pred + deep_pred)

            return torch.cat(preds_list, dim=1)

        else:
            # Case 'sampled': process given item_indices
            if item_indices.dim() == 1:
                item_indices = item_indices.unsqueeze(1)

            pad_seq = item_indices.size(1)

            # Get Item Embeddings and Bias
            item_emb = self.item_embedding(item_indices)
            item_b = self.item_bias(item_indices).squeeze(-1)

            # Get Feature Embeddings and Bias
            feat_emb_list = get_feature_embeddings(
                item_indices
            )  # List of [batch, pad_seq, emb]
            feat_b = get_feature_biases(item_indices)  # [batch, pad_seq]

            # Linear Part
            linear_pred = fixed_linear.unsqueeze(1) + item_b + feat_b

            # FM Part
            # Aggregate Item + Features
            item_feat_sum = item_emb
            item_feat_sq_sum = item_emb.pow(2)

            for f_emb in feat_emb_list:
                item_feat_sum = item_feat_sum + f_emb
                item_feat_sq_sum = item_feat_sq_sum + f_emb.pow(2)

            sum_v_fixed_exp = sum_v_fixed.unsqueeze(1)
            sum_sq_v_fixed_exp = sum_sq_v_fixed.unsqueeze(1)

            sum_v_total_sq = (sum_v_fixed_exp + item_feat_sum).pow(2)
            sum_sq_total = sum_sq_v_fixed_exp + item_feat_sq_sum

            fm_pred = 0.5 * (sum_v_total_sq - sum_sq_total).sum(dim=2)

            # Deep Part
            user_emb_exp = user_emb.unsqueeze(1).expand(-1, pad_seq, -1)
            ctx_emb_exp_list = [
                c.unsqueeze(1).expand(-1, pad_seq, -1) for c in ctx_emb_list
            ]

            deep_input_block = torch.cat(
                [user_emb_exp, item_emb] + feat_emb_list + ctx_emb_exp_list, dim=2
            )
            deep_input_flat = deep_input_block.view(
                -1, self.num_fields * self.embedding_size
            )

            deep_out = self.mlp_layers(deep_input_flat)
            deep_pred = self.deep_predict_layer(deep_out).view(batch_size, pad_seq)

            return linear_pred + fm_pred + deep_pred
