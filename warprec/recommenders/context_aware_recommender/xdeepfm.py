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


class CIN(nn.Module):
    """Compressed Interaction Network (CIN) module."""

    def __init__(
        self,
        num_fields: int,
        embedding_size: int,
        cin_layer_size: List[int],
        direct: bool = False,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_size = embedding_size
        self.cin_layer_size = cin_layer_size
        self.direct = direct

        # If not direct, layer sizes must be even to allow splitting
        if not self.direct:
            self.cin_layer_size = [int(x // 2 * 2) for x in cin_layer_size]

        self.conv1d_list = nn.ModuleList()
        self.field_nums = [self.num_fields]

        for _, layer_size in enumerate(self.cin_layer_size):
            # Input channels for Conv1d is H_{k-1} * m
            in_channels = self.field_nums[-1] * self.field_nums[0]

            # Conv1d acts as the filter W^{k,h} sliding over the embedding dimension D
            self.conv1d_list.append(nn.Conv1d(in_channels, layer_size, 1))

            if self.direct:
                self.field_nums.append(layer_size)
            else:
                self.field_nums.append(layer_size // 2)

        # Calculate final output dimension (sum of pooled vectors)
        if self.direct:
            self.final_len = sum(self.cin_layer_size)
        else:
            self.final_len = (
                sum(self.cin_layer_size[:-1]) // 2 + self.cin_layer_size[-1]
            )

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, num_fields, embedding_size]
        batch_size = x.shape[0]
        hidden_nn_layers = [x]
        final_result = []

        for i, layer_size in enumerate(self.cin_layer_size):
            # Outer product (Interaction): X^{k-1} * X^0
            # Tensor shape: [batch_size, H_{k-1}, m, embedding_size]
            z_i = torch.einsum(
                "bhd,bmd->bhmd", hidden_nn_layers[-1], hidden_nn_layers[0]
            )

            # Reshape for Conv1d (Compression)
            # Flatten H_{k-1} and m into the channel dimension
            # Tensor shape: [batch_size, H_{k-1} * m, embedding_size]
            z_i = z_i.view(
                batch_size, self.field_nums[0] * self.field_nums[i], self.embedding_size
            )

            # Convolution and Activation
            # Tensor shape: [batch_size, H_k, embedding_size]
            z_i = self.conv1d_list[i](z_i)
            output = torch.relu(z_i)

            # Split or Direct (Architecture choice)
            if self.direct:
                direct_connect = output
                next_hidden = output

                # In direct mode, we always append to hidden layers
                hidden_nn_layers.append(next_hidden)
            else:
                if i != len(self.cin_layer_size) - 1:
                    # Split: half for next hidden layer, half for final pooling
                    next_hidden, direct_connect = torch.split(
                        output, 2 * [layer_size // 2], 1
                    )

                    # Append only if there is a next iteration
                    hidden_nn_layers.append(next_hidden)
                else:
                    # Last layer: everything goes to pooling
                    direct_connect = output

            final_result.append(direct_connect)

        # Concatenate and Sum Pooling
        # [batch_size, sum(H_k), embedding_size]
        result = torch.cat(final_result, dim=1)

        return torch.sum(result, dim=-1)  # [batch_size, sum(H_k)]


@model_registry.register(name="xDeepFM")
class xDeepFM(ContextRecommenderUtils, IterativeRecommender):
    """Implementation of xDeepFM algorithm from
        xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, KDD 2018.

    For further details, check the `paper <https://arxiv.org/abs/1803.05170>`_.

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
        cin_layer_size (List[int]): The size of CIN layers.
        dropout (float): The dropout probability.
        direct (bool): The type of output of CIN module.
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
    cin_layer_size: List[int]
    dropout: float
    direct: bool
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
        self.chunk_size = kwargs.get("chunk_size", 4096)
        self.mlp_hidden_size = list(self.mlp_hidden_size)
        self.cin_layer_size = list(self.cin_layer_size)
        self.num_fields = 2 + len(self.feature_labels) + len(self.context_labels)

        # CIN (Compressed Interaction Network) - Explicit High-order
        self.cin = CIN(
            self.num_fields, self.embedding_size, self.cin_layer_size, self.direct
        )
        self.cin_linear = nn.Linear(self.cin.final_len, 1)

        # DNN (MLP) - Implicit High-order
        input_dim = self.num_fields * self.embedding_size
        self.mlp_layers = MLP([input_dim] + self.mlp_hidden_size, self.dropout)
        self.dnn_linear = nn.Linear(self.mlp_hidden_size[-1], 1)

        # Losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = EmbLoss()

        # Initialize weights
        self.apply(self._init_weights)

    def train_step(self, batch: Any, *args, **kwargs) -> Tensor:
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

        # [batch_size, num_fields, embedding_size]
        stacked_embeddings = torch.stack(embeddings_list, dim=1)

        # CIN Part
        cin_output = self.cin(stacked_embeddings)
        cin_score = self.cin_linear(cin_output).squeeze(-1)

        # DNN Part
        batch_size = stacked_embeddings.shape[0]
        dnn_input = stacked_embeddings.view(batch_size, -1)
        dnn_output = self.mlp_layers(dnn_input)
        dnn_score = self.dnn_linear(dnn_output).squeeze(-1)

        # Final Sum
        return linear_part + cin_score + dnn_score

    def _compute_network_scores(
        self,
        u_emb: Tensor,
        i_emb: Tensor,
        feat_emb_list: List[Tensor],
        ctx_emb_list: List[Tensor],
        batch_size: int,
        num_items: int,
    ) -> Tensor:
        """Compute scores of deep part (CIN + MLP) efficiently"""
        total_rows = batch_size * num_items

        # Create memory efficient view
        u_view = u_emb.unsqueeze(1).expand(-1, num_items, -1).reshape(total_rows, -1)

        # Handle Feature views
        feat_views = []
        for f in feat_emb_list:
            f_exp = f.unsqueeze(0).expand(batch_size, -1, -1).reshape(total_rows, -1)
            feat_views.append(f_exp)

        # Handle Context views
        ctx_views = []
        for c in ctx_emb_list:
            ctx_views.append(
                c.unsqueeze(1).expand(-1, num_items, -1).reshape(total_rows, -1)
            )

        i_view = i_emb.reshape(total_rows, -1)

        # Pre-allocate tensor to memory to avoid overhead later
        all_scores = torch.empty(total_rows, device=self.device)

        # Loop on chunk size parameter
        for start in range(0, total_rows, self.chunk_size):
            end = min(start + self.chunk_size, total_rows)

            # Slice the views
            u_chunk = u_view[start:end]
            i_chunk = i_view[start:end]
            f_chunks = [f[start:end] for f in feat_views]
            c_chunks = [c[start:end] for c in ctx_views]

            # Materialize ONLY the chunk
            # NOTE: This will actually use memory
            chunk_stack = torch.stack([u_chunk, i_chunk] + f_chunks + c_chunks, dim=1)

            # Forward CIN
            cin_out = self.cin(chunk_stack)
            cin_s = self.cin_linear(cin_out).squeeze(-1)

            # Forward MLP
            dnn_in = chunk_stack.view(chunk_stack.size(0), -1)
            dnn_out = self.mlp_layers(dnn_in)
            dnn_s = self.dnn_linear(dnn_out).squeeze(-1)

            # Save in place in the pre-allocated tensor
            all_scores[start:end] = cin_s + dnn_s

        return all_scores.view(batch_size, num_items)

    @torch.no_grad()
    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        contexts: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction using the xDeepFM model.

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

        # Linear Parts (User + Context)
        fixed_linear = self.global_bias + self.user_bias(user_indices).squeeze(-1)
        if contexts is not None and self.context_labels:
            for idx, name in enumerate(self.context_labels):
                fixed_linear += self.context_bias[name](contexts[:, idx]).squeeze(-1)

        # Embeddings (User + Context)
        u_emb = self.user_embedding(user_indices)
        ctx_emb_list = []
        if contexts is not None and self.context_labels:
            ctx_emb_list = [
                self.context_embedding[name](contexts[:, idx])
                for idx, name in enumerate(self.context_labels)
            ]

        if item_indices is None:
            # Case 'full': iterate through all items in memory-safe blocks
            preds_list = []

            for start in range(0, self.n_items, self.block_size):
                end = min(start + self.block_size, self.n_items)
                current_block_len = end - start

                # Indices and Embeddings for this block
                items_block = torch.arange(start, end, device=self.device)

                # [block_size, embedding_size]
                item_emb_block = self.item_embedding(items_block)
                item_bias_block = self.item_bias(items_block).squeeze(-1)

                # Retrieve block feature embeddings and bias
                feat_emb_block_list = self._get_feature_embeddings(items_block)
                feat_bias_block = self._get_feature_bias(items_block)

                # Linear Part
                item_bias_block = self.item_bias(items_block).squeeze(-1)
                linear_pred = (
                    fixed_linear.unsqueeze(1)
                    + item_bias_block.unsqueeze(0)
                    + feat_bias_block.unsqueeze(0)
                )

                # Expand item part
                item_emb_expanded = item_emb_block.unsqueeze(0).expand(
                    batch_size, -1, -1
                )  # [batch_size, block_size, embedding_size]

                # Compute scores efficiently
                net_scores = self._compute_network_scores(
                    u_emb,
                    item_emb_expanded,
                    feat_emb_block_list,
                    ctx_emb_list,
                    batch_size,
                    current_block_len,
                )

                preds_list.append(linear_pred + net_scores)

            return torch.cat(preds_list, dim=1)

        else:
            # Case 'sampled': process given item_indices
            pad_seq = item_indices.size(1)

            item_emb = self.item_embedding(item_indices)
            item_bias = self.item_bias(item_indices).squeeze(-1)

            # Retrieve item feature embeddings & bias
            feat_emb_list = self._get_feature_embeddings(item_indices)
            feat_bias = self._get_feature_bias(item_indices)

            # Linear
            item_bias = self.item_bias(item_indices).squeeze(-1)
            linear_pred = fixed_linear.unsqueeze(1) + item_bias + feat_bias

            # Build the stack for xDeepFM
            u_emb_exp = u_emb.unsqueeze(1).expand(-1, pad_seq, -1)
            ctx_emb_exp_list = [
                c.unsqueeze(1).expand(-1, pad_seq, -1) for c in ctx_emb_list
            ]

            # Stack consistent with forward: User, Item, Features, Contexts
            stack = torch.stack(
                [u_emb_exp, item_emb] + feat_emb_list + ctx_emb_exp_list, dim=2
            )

            # Flatten to process the whole batch together
            total_rows = batch_size * pad_seq
            stack_flat = stack.view(total_rows, self.num_fields, self.embedding_size)

            # Forward CIN
            cin_out = self.cin(stack_flat)
            cin_s = self.cin_linear(cin_out).squeeze(-1)

            # Forward MLP
            dnn_in = stack_flat.view(total_rows, -1)
            dnn_out = self.mlp_layers(dnn_in)
            dnn_s = self.dnn_linear(dnn_out).squeeze(-1)

            net_scores = (cin_s + dnn_s).view(batch_size, pad_seq)

            return linear_pred + net_scores
