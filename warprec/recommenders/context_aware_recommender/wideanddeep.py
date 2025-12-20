import torch
from torch import nn, Tensor
from typing import Any, Optional, List

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    ContextRecommenderUtils,
)
from warprec.recommenders.layers import MLP
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


@model_registry.register(name="WideAndDeep")
class WideAndDeep(ContextRecommenderUtils, IterativeRecommender):
    """Implementation of Wide & Deep algorithm from
        Wide & Deep Learning for Recommender Systems, DLRS 2016.

    For further details, check the `paper <https://arxiv.org/abs/1606.07792>`_.

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

        # Deep Part (DNN)
        self.num_fields = 2 + len(self.context_labels)

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
        user, item, rating, contexts = batch

        prediction = self.forward(user, item, contexts)

        # Compute BCE loss
        loss = self.bce_loss(prediction, rating)

        # Compute L2 regularization on embeddings and biases
        reg_params = self.get_reg_params(user, item, contexts)
        reg_loss = self.reg_weight * self.reg_loss(*reg_params)

        return loss + reg_loss

    def forward(self, user: Tensor, item: Tensor, contexts: Tensor) -> Tensor:
        """Forward pass of the WideDeep model.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.
            contexts (Tensor): The tensor containing the context of the interactions.

        Returns:
            Tensor: The prediction score for each triplet (user, item, context).
        """
        # Wide Part (Linear)
        wide_part = self.compute_first_order(user, item, contexts)

        # Deep Part (DNN)
        embeddings_list = [self.user_embedding(user), self.item_embedding(item)]
        for idx, name in enumerate(self.context_labels):
            embeddings_list.append(self.context_embedding[name](contexts[:, idx]))

        stacked_embeddings = torch.stack(embeddings_list, dim=1)
        batch_size = stacked_embeddings.shape[0]

        deep_input = stacked_embeddings.view(batch_size, -1)
        deep_output = self.mlp_layers(deep_input)
        deep_part = self.deep_predict_layer(deep_output).squeeze(-1)

        return wide_part + deep_part

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the WideAndDeep model.

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

        # Wide Fixed
        fixed_wide = self.global_bias + self.user_bias(user_indices).squeeze(-1)

        # Deep Fixed Parts (List of embeddings to be concatenated)
        # Order must match forward: User, Item, Contexts
        user_emb = self.user_embedding(user_indices)  # [batch_size, embedding_size]
        ctx_emb_list = []

        # Process Contexts
        for idx, name in enumerate(self.context_labels):
            ctx_input = contexts[:, idx]

            # Wide
            fixed_wide += self.context_bias[name](ctx_input).squeeze(-1)

            # Deep
            ctx_emb = self.context_embedding[name](ctx_input)
            ctx_emb_list.append(ctx_emb)

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

                # Wide Part
                wide_pred = fixed_wide.unsqueeze(1) + item_b.unsqueeze(0)

                # Deep Part
                # We need to construct [User, Item, Contexts] for every pair
                user_emb_exp = user_emb.unsqueeze(1).expand(-1, current_block_size, -1)
                item_emb_exp = item_emb.unsqueeze(0).expand(batch_size, -1, -1)

                # Expand Contexts
                ctx_emb_exp_list = [
                    c.unsqueeze(1).expand(-1, current_block_size, -1)
                    for c in ctx_emb_list
                ]

                # Concatenate in order: User, Item, Contexts
                deep_input_block = torch.cat(
                    [user_emb_exp, item_emb_exp] + ctx_emb_exp_list, dim=2
                )  # [batch_size, block_size, num_fields * embedding_size]

                # Flatten for MLP
                deep_input_flat = deep_input_block.view(
                    -1, self.num_fields * self.embedding_size
                )  # [batch_size * block_size, input_dim]

                deep_out = self.mlp_layers(deep_input_flat)
                deep_pred = self.deep_predict_layer(deep_out).view(
                    batch_size, current_block_size
                )

                # Sum and append
                preds_list.append(wide_pred + deep_pred)

            return torch.cat(preds_list, dim=1)

        else:
            # Case 'sampled': process given item_indices
            if item_indices.dim() == 1:
                item_indices = item_indices.unsqueeze(1)  # [batch_size, 1]

            pad_seq = item_indices.size(1)

            # Get Item Embeddings and Bias
            item_emb = self.item_embedding(
                item_indices
            )  # [batch_size, pad_seq, embedding_size]
            item_b = self.item_bias(item_indices).squeeze(-1)  # [batch_size, pad_seq]

            # Wide Part
            wide_pred = fixed_wide.unsqueeze(1) + item_b

            # Deep Part
            user_emb_exp = user_emb.unsqueeze(1).expand(-1, pad_seq, -1)

            # Expand Contexts
            ctx_emb_exp_list = [
                c.unsqueeze(1).expand(-1, pad_seq, -1) for c in ctx_emb_list
            ]

            # Concatenate: User, Item, Contexts
            deep_input_block = torch.cat(
                [user_emb_exp, item_emb] + ctx_emb_exp_list, dim=2
            )

            deep_input_flat = deep_input_block.view(
                -1, self.num_fields * self.embedding_size
            )

            deep_out = self.mlp_layers(deep_input_flat)
            deep_pred = self.deep_predict_layer(deep_out).view(batch_size, pad_seq)

            return wide_pred + deep_pred
