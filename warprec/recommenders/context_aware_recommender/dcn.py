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


@model_registry.register(name="DCN")
class DCN(ContextRecommenderUtils, IterativeRecommender):
    """Implementation of Deep & Cross Network (DCN) from
        Deep & Cross Network for Ad Click Predictions, ADKDD 2017.

    For further details, check the `paper <https://arxiv.org/abs/1708.05123>`_.

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
        cross_layer_num (int): The number of cross layers.
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
    cross_layer_num: int
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

        self.block_size = kwargs.get("block_size", 50)
        self.mlp_hidden_size = list(self.mlp_hidden_size)

        # DCN Specific Layers
        self.num_fields = 2 + len(self.context_labels)
        self.input_dim = self.num_fields * self.embedding_size

        # Cross Network Parameters
        # Weights and Biases for each layer
        self.cross_layer_w = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.input_dim))
                for _ in range(self.cross_layer_num)
            ]
        )
        self.cross_layer_b = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.input_dim))
                for _ in range(self.cross_layer_num)
            ]
        )

        # Deep Network (MLP)
        # Input is the flattened embedding vector
        self.mlp_layers = MLP([self.input_dim] + self.mlp_hidden_size, self.dropout)

        # Final Prediction Layer
        # Input: Output of Cross Network + Output of Deep Network
        # Cross Network output size is same as input_dim
        final_dim = self.input_dim + self.mlp_hidden_size[-1]
        self.predict_layer = nn.Linear(final_dim, 1)

        # Losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = EmbLoss()

        # Initialize weights
        self.apply(self._init_weights)

    def _cross_network(self, x_0: Tensor) -> Tensor:
        """Computes the output of the Cross Network.

        Formula: x_{l+1} = x_0 * (x_l^T * w_l) + b_l + x_l
        """
        x_l = x_0
        for i in range(self.cross_layer_num):
            # x_l: [batch, input_dim]
            # w: [input_dim]
            # x_l^T * w -> dot product per sample -> [batch, 1]
            # We use matmul for efficiency: (x_l @ w)

            # [batch, 1]
            xl_w = torch.matmul(x_l, self.cross_layer_w[i]).unsqueeze(1)

            # x_0 * scalar + bias + x_l
            x_l = x_0 * xl_w + self.cross_layer_b[i] + x_l

        return x_l

    def _compute_logits(self, dcn_input: Tensor) -> Tensor:
        """Core logic of DCN: Shared between forward and predict.

        Args:
            dcn_input (Tensor): Flattened input embeddings [batch_size, input_dim]

        Returns:
            Tensor: Logits [batch_size, 1]
        """
        # Deep Part
        deep_output = self.mlp_layers(dcn_input)

        # Cross Part
        cross_output = self._cross_network(dcn_input)

        # Stack and Predict
        stack = torch.cat([cross_output, deep_output], dim=-1)
        output = self.predict_layer(stack)

        return output

    def train_step(self, batch: Any, *args, **kwargs) -> Tensor:
        user, item, rating, contexts = batch

        prediction = self.forward(user, item, contexts)

        # Compute BCE loss
        loss = self.bce_loss(prediction, rating)

        # Compute L2 regularization on embeddings and biases
        reg_params = self.get_reg_params(user, item, contexts)
        reg_loss = self.reg_weight * self.reg_loss(*reg_params)

        return loss + reg_loss

    def forward(self, user: Tensor, item: Tensor, contexts: Tensor) -> Tensor:
        """Forward pass of the DCN model.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.
            contexts (Tensor): The tensor containing the context of the interactions.

        Returns:
            Tensor: The prediction score for each triplet (user, item, context).
        """
        # Retrieve Embeddings
        u_emb = self.user_embedding(user)
        i_emb = self.item_embedding(item)

        # Contexts: List of [batch, emb]
        ctx_emb_list = [
            self.context_embedding[name](contexts[:, idx])
            for idx, name in enumerate(self.context_labels)
        ]

        # Stack and Flatten
        embeddings_list = [u_emb, i_emb] + ctx_emb_list
        dcn_input = torch.cat(
            embeddings_list, dim=1
        )  # [batch, num_fields * embedding_size]

        # Compute Network
        output = self._compute_logits(dcn_input)

        return output.squeeze(-1)

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the DCN model.

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

        # Retrieve Fixed Embeddings (User + Contexts)
        # [batch, embedding_size]
        user_emb = self.user_embedding(user_indices)
        ctx_emb_list = [
            self.context_embedding[name](contexts[:, idx])
            for idx, name in enumerate(self.context_labels)
        ]

        # Helper function to process item block
        def process_block(items_emb_block: Tensor) -> Tensor:
            n_items = items_emb_block.shape[-2]

            # Expand User & Contexts to match items dimension
            u_exp = user_emb.unsqueeze(1).expand(-1, n_items, -1)
            c_exp_list = [c.unsqueeze(1).expand(-1, n_items, -1) for c in ctx_emb_list]

            # Handle Item Embedding expansion if necessary
            if items_emb_block.dim() == 2:
                # Case: Full prediction (items shared across all users in batch)
                # [n_items, emb] -> [batch, n_items, emb]
                i_exp = items_emb_block.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # Case: Sampled prediction (specific items per user)
                i_exp = items_emb_block

            # Concatenate all fields
            dcn_input_block = torch.cat([u_exp, i_exp] + c_exp_list, dim=2)

            # Flatten for the network: [batch_size * n_items, input_dim]
            dcn_input_flat = dcn_input_block.view(-1, self.input_dim)

            # Compute Logits
            logits = self._compute_logits(dcn_input_flat)

            # Reshape back: [batch_size, n_items]
            return logits.view(batch_size, n_items)

        if item_indices is None:
            # Case 'full': iterate through all items in memory-safe blocks
            preds_list = []
            for start in range(0, self.n_items, self.block_size):
                end = min(start + self.block_size, self.n_items)

                # Get item embeddings for the block (shared for all users)
                items_block = torch.arange(start, end, device=self.device)
                item_emb_block = self.item_embedding(
                    items_block
                )  # [block_size, embedding_size]

                # Process the block
                preds_list.append(process_block(item_emb_block))

            return torch.cat(preds_list, dim=1)

        else:
            # Case 'sampled': process given item_indices
            item_emb = self.item_embedding(
                item_indices
            )  # [batch_size, seq_len, embedding_size]
            return process_block(item_emb)
