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


@model_registry.register(name="NFM")
class NFM(ContextRecommenderUtils, IterativeRecommender):
    """Implementation of NFM algorithm from
        Neural Factorization Machines for Sparse Predictive Analytics, SIGIR 2017.

    For further details, check the `paper <https://arxiv.org/abs/1708.05027>`_.

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
        user, item, rating, contexts = batch

        prediction = self.forward(user, item, contexts)

        # Compute BCE loss
        loss = self.bce_loss(prediction, rating)

        # Compute L2 regularization on embeddings and biases
        reg_params = self.get_reg_params(user, item, contexts)
        reg_loss = self.reg_weight * self.reg_loss(*reg_params)

        return loss + reg_loss

    def forward(self, user: Tensor, item: Tensor, contexts: Tensor) -> Tensor:
        """Forward pass of the NFM model.

        Args:
            user (Tensor): The tensor containing the user indexes.
            item (Tensor): The tensor containing the item indexes.
            contexts (Tensor): The tensor containing the context of the interactions.

        Returns:
            Tensor: The prediction score for each triplet (user, item, context).
        """
        # Linear Part
        linear_part = self.compute_first_order(user, item, contexts)

        # Second Order (Interaction Part)
        embeddings_list = [self.user_embedding(user), self.item_embedding(item)]
        for idx, name in enumerate(self.context_labels):
            embeddings_list.append(self.context_embedding[name](contexts[:, idx]))

        fm_input = torch.stack(embeddings_list, dim=1)

        # Bi-Interaction Pooling
        sum_of_vectors = torch.sum(fm_input, dim=1)
        sum_of_squares = torch.sum(fm_input.pow(2), dim=1)
        bi_interaction = 0.5 * (sum_of_vectors.pow(2) - sum_of_squares)

        # Neural Layers
        bi_interaction = self.batch_norm(bi_interaction)
        mlp_output = self.mlp_layers(bi_interaction)
        prediction_score = self.predict_layer(mlp_output).squeeze(-1)

        return linear_part + prediction_score

    @torch.no_grad()
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

        # Process Contexts
        for idx, name in enumerate(self.context_labels):
            ctx_input = contexts[:, idx]

            # Linear
            fixed_linear += self.context_bias[name](ctx_input).squeeze(-1)

            # Interaction
            ctx_emb = self.context_embedding[name](ctx_input)
            sum_v_fixed += ctx_emb
            sum_sq_v_fixed += ctx_emb.pow(2)

        if item_indices is None:
            # Case 'full': iterate through all items in memory-safe blocks
            preds_list = []

            for start in range(0, self.n_items, self.block_size):
                end = min(start + self.block_size, self.n_items)
                current_block_size = end - start

                # Item indices for this block
                items_block = torch.arange(start, end, device=user_indices.device)

                # Get Item Embeddings & Bias
                item_emb = self.item_embedding(
                    items_block
                )  # [block_size, embedding_size]
                item_b = self.item_bias(items_block).squeeze(-1)  # [block_size]

                # Calculate Bi-Interaction for the block
                # (V_fixed + V_item)^2
                sum_v_total = sum_v_fixed.unsqueeze(1) + item_emb.unsqueeze(0)
                sum_v_total_sq = sum_v_total.pow(2)

                # (V_fixed^2 + V_item^2)
                sum_sq_total = sum_sq_v_fixed.unsqueeze(1) + item_emb.pow(2).unsqueeze(
                    0
                )

                # Interaction vector: [batch_size, block_size, embedding_size]
                bi_interaction = 0.5 * (sum_v_total_sq - sum_sq_total)

                # Flatten for MLP: [batch_size * block_size, embedding_size]
                bi_interaction_flat = bi_interaction.view(-1, self.embedding_size)

                # Pass through Neural Part
                bi_interaction_flat = self.batch_norm(bi_interaction_flat)
                mlp_out = self.mlp_layers(bi_interaction_flat)
                neural_pred = self.predict_layer(mlp_out).view(
                    batch_size, current_block_size
                )

                # Calculate Linear Part
                linear_pred = fixed_linear.unsqueeze(1) + item_b.unsqueeze(
                    0
                )  # [batch_size, block_size]

                # Sum and append
                preds_list.append(linear_pred + neural_pred)

            return torch.cat(preds_list, dim=1)

        else:
            # Case 'sampled': process given item_indices
            if item_indices.dim() == 1:
                item_indices = item_indices.unsqueeze(1)  # [batch_size, 1]

            pad_seq = item_indices.size(1)

            # Get Item Embeddings & Bias
            item_emb = self.item_embedding(
                item_indices
            )  # [batch_size, pad_seq, embedding_size]
            item_b = self.item_bias(item_indices).squeeze(-1)  # [batch_size, pad_seq]

            # Calculate Bi-Interaction
            sum_v_fixed_exp = sum_v_fixed.unsqueeze(1)
            sum_sq_v_fixed_exp = sum_sq_v_fixed.unsqueeze(1)

            sum_v_total_sq = (sum_v_fixed_exp + item_emb).pow(2)
            sum_sq_total = sum_sq_v_fixed_exp + item_emb.pow(2)

            bi_interaction = 0.5 * (sum_v_total_sq - sum_sq_total)

            # Flatten for MLP
            bi_interaction_flat = bi_interaction.view(-1, self.embedding_size)

            # Neural Part
            bi_interaction_flat = self.batch_norm(bi_interaction_flat)
            mlp_out = self.mlp_layers(bi_interaction_flat)
            neural_pred = self.predict_layer(mlp_out).view(batch_size, pad_seq)

            # Linear Part
            linear_pred = fixed_linear.unsqueeze(1) + item_b

            return linear_pred + neural_pred
