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


@model_registry.register(name="DCNv2")
class DCNv2(ContextRecommenderUtils, IterativeRecommender):
    """Implementation of Deep & Cross Network V2 (DCNv2) from
        Dcn v2: Improved deep & cross network and practical lessons for web-scale, WWW 2021.

    For further details, check the `paper <https://arxiv.org/abs/2008.13535>`_.

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
        model_structure (str): The model structure to use.
        use_mixed (bool): Wether or not use the MoE.
        expert_num (int): The number of expert to use in MoE.
        low_rank (int): The low rank dimension.
        reg_weight (float): The L2 regularization weight.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): Number of negative samples for training.

    Raises:
        ValueError: If model_structure parameter is not supported.
    """

    DATALOADER_TYPE = DataLoaderType.ITEM_RATING_LOADER_WITH_CONTEXT

    embedding_size: int
    mlp_hidden_size: List[int]
    cross_layer_num: int
    dropout: float
    model_structure: str
    use_mixed: bool
    expert_num: int
    low_rank: int
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

        # Input Dimensions
        self.num_fields = 2 + len(self.context_labels)
        self.input_dim = self.num_fields * self.embedding_size

        if self.use_mixed:
            # Mixed Cross Network (MoE + Low Rank)
            # U: [Layers, Experts, Input, Rank]
            self.cross_u = nn.Parameter(
                torch.randn(
                    self.cross_layer_num, self.expert_num, self.input_dim, self.low_rank
                )
            )
            # V: [Layers, Experts, Input, Rank]
            self.cross_v = nn.Parameter(
                torch.randn(
                    self.cross_layer_num, self.expert_num, self.input_dim, self.low_rank
                )
            )
            # C: [Layers, Experts, Rank, Rank]
            self.cross_c = nn.Parameter(
                torch.randn(
                    self.cross_layer_num, self.expert_num, self.low_rank, self.low_rank
                )
            )

            # Gating: [Layers, Input, Experts] -> Linear transformation per layer
            self.gating = nn.ModuleList(
                [
                    nn.Linear(self.input_dim, self.expert_num)
                    for _ in range(self.cross_layer_num)
                ]
            )
        else:
            # Standard DCNv2 Matrix Cross Network
            # W: [Layers, Input, Input] -> Full Matrix (Expensive for high dim)
            self.cross_w = nn.Parameter(
                torch.randn(self.cross_layer_num, self.input_dim, self.input_dim)
            )

        # Bias: [Layers, Input]
        self.cross_b = nn.Parameter(torch.zeros(self.cross_layer_num, self.input_dim))

        # Deep Network (MLP)
        self.mlp_layers = MLP([self.input_dim] + self.mlp_hidden_size, self.dropout)

        # Prediction Layer
        if self.model_structure == "parallel":
            final_dim = self.input_dim + self.mlp_hidden_size[-1]
        elif self.model_structure == "stacked":
            final_dim = self.mlp_hidden_size[-1]
        else:
            raise ValueError(
                f"Model structure {self.model_structure} not supported. "
                "Model structure supported are 'parallel' and 'stacked'."
            )

        self.predict_layer = nn.Linear(final_dim, 1)

        # Losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = EmbLoss()

        # Initialize weights
        self.apply(self._init_weights)

    def _cross_network_matrix(self, x_0: Tensor) -> Tensor:
        """Standard DCNv2 (Matrix).

        Equation: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
        """
        x_l = x_0
        for i in range(self.cross_layer_num):
            # W * x_l -> [batch_size, input_dim]
            # Using linear transformation logic: x @ W.T

            # [batch_size, input_dim] @ [input_dim, input_dim] -> [batch_size, input_dim]
            wl_xl = torch.matmul(x_l, self.cross_w[i])

            # Add bias
            wl_xl = wl_xl + self.cross_b[i]

            # Element-wise product with x_0 and add residual
            x_l = x_0 * wl_xl + x_l

        return x_l

    def _cross_network_mixed(self, x_0: Tensor) -> Tensor:
        """Mixed DCNv2 (MoE + Low Rank).

        Vectorized implementation using einsum to avoid loops over experts.
        """
        x_l = x_0  # [batch_size, input_dim]

        for i in range(self.cross_layer_num):
            # Gating
            # [batch_size, expert_num]
            gating_score = self.gating[i](x_l)
            gating_prob = torch.softmax(gating_score, dim=1)

            # Expert Computation (Low Rank)
            # We want to compute: U * tanh(C * tanh(V^T * x))
            xl_v = torch.einsum("bi, eir -> ber", x_l, self.cross_v[i])
            xl_v = torch.tanh(xl_v)  # [batch_size, expert_num, low_rank]

            # Mix in Low Rank (C * result)
            xl_c = torch.einsum("ber, err -> ber", xl_v, self.cross_c[i])
            xl_c = torch.tanh(xl_c)  # [batch_size, expert_num, low_rank]

            # Project back to High Rank (U * result)
            # [batch_size, expert_num, input_dim]
            expert_outputs = torch.einsum("ber, eir -> bei", xl_c, self.cross_u[i])

            # Add bias (broadcast over experts)
            expert_outputs = expert_outputs + self.cross_b[i].unsqueeze(0).unsqueeze(0)

            # Element-wise with x_0: x_0 * (Expert_Out)
            expert_outputs = x_0.unsqueeze(1) * expert_outputs

            # Weighted Sum of Experts (MoE)
            # Gating: [batch_size, expert_num] -> [batch_size, expert_num, 1]
            # Output: [batch_size, input_dim]
            moe_output = torch.sum(expert_outputs * gating_prob.unsqueeze(-1), dim=1)

            # Residual
            x_l = moe_output + x_l

        return x_l

    def _compute_logits(self, dcn_input: Tensor) -> Tensor:
        """Core logic shared between forward and predict."""

        # Cross Network
        if self.use_mixed:
            cross_output = self._cross_network_mixed(dcn_input)
        else:
            cross_output = self._cross_network_matrix(dcn_input)

        # Deep Network
        if self.model_structure == "parallel":
            deep_output = self.mlp_layers(dcn_input)
            stack = torch.cat([cross_output, deep_output], dim=-1)
        else:  # stacked
            # Deep network takes cross output as input
            deep_output = self.mlp_layers(cross_output)
            stack = deep_output

        # 3. Prediction
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
        # Retrieve Embeddings
        u_emb = self.user_embedding(user)
        i_emb = self.item_embedding(item)
        ctx_emb_list = [
            self.context_embedding[name](contexts[:, idx])
            for idx, name in enumerate(self.context_labels)
        ]

        # Stack and Flatten
        embeddings_list = [u_emb, i_emb] + ctx_emb_list
        dcn_input = torch.cat(embeddings_list, dim=1)

        # Compute
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
        """Prediction using the DCNv2 model.

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

        # Retrieve Fixed Embeddings
        user_emb = self.user_embedding(user_indices)
        ctx_emb_list = [
            self.context_embedding[name](contexts[:, idx])
            for idx, name in enumerate(self.context_labels)
        ]

        # Helper for block processing
        def process_block(items_emb_block: Tensor) -> Tensor:
            num_items = items_emb_block.shape[-2]

            # Expand User and Contexts
            u_exp = user_emb.unsqueeze(1).expand(-1, num_items, -1)
            c_exp_list = [
                c.unsqueeze(1).expand(-1, num_items, -1) for c in ctx_emb_list
            ]

            # Handle Item Embedding
            if items_emb_block.dim() == 2:  # Full
                i_exp = items_emb_block.unsqueeze(0).expand(batch_size, -1, -1)
            else:  # Sampled
                i_exp = items_emb_block

            # Concatenate
            dcn_input_block = torch.cat([u_exp, i_exp] + c_exp_list, dim=2)
            dcn_input_flat = dcn_input_block.view(-1, self.input_dim)

            # Compute
            logits = self._compute_logits(dcn_input_flat)

            return logits.view(batch_size, num_items)

        if item_indices is None:
            # Case 'full': iterate through all items in memory-safe blocks
            preds_list = []
            for start in range(0, self.n_items, self.block_size):
                end = min(start + self.block_size, self.n_items)
                items_block = torch.arange(start, end, device=self.device)
                item_emb_block = self.item_embedding(items_block)
                preds_list.append(process_block(item_emb_block))
            return torch.cat(preds_list, dim=1)
        else:
            # Case 'sampled': process given item_indices
            item_emb = self.item_embedding(item_indices)
            return process_block(item_emb)
