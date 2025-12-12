import torch
from torch import nn, Tensor
from typing import Any, Optional, List

from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    ContextRecommenderUtils,
)
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.data.entities import Interactions
from warprec.utils.registry import model_registry


class AttentionLayer(nn.Module):
    """Implements the Attention Network.

    Equation: a_ij = h^T ReLU(W(v_i * v_j) + b)
    """

    def __init__(self, embedding_size: int, attention_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, attention_size),
            nn.ReLU(),
            nn.Linear(attention_size, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch_size, num_pairs, embedding_size]
        # scores: [batch_size, num_pairs, 1]
        logits = self.mlp(x)
        return torch.softmax(logits, dim=1)


@model_registry.register(name="AFM")
class AFM(ContextRecommenderUtils, IterativeRecommender):
    """Implementation of AFM algorithm from
        Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks, IJCAI 2017.

    For further details, check the `paper <https://arxiv.org/abs/1708.04617>`_.

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
        attention_size (int): The size of the attention network hidden layer.
        dropout (float): The dropout probability.
        eval_chunk_size (int): The chunk size to use in evaluation.
        reg_weight (float): The L2 regularization weight for embeddings.
        weight_decay (float): The value of weight decay used in the optimizer.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        neg_samples (int): Number of negative samples for training.
    """

    DATALOADER_TYPE = DataLoaderType.ITEM_RATING_LOADER_WITH_CONTEXT

    embedding_size: int
    attention_size: int
    dropout: float
    eval_chunk_size: int
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

        # Attention Network
        self.attention_layer = AttentionLayer(self.embedding_size, self.attention_size)

        # Projection Vector p
        # Weights the final pooled vector to produce the score
        self.p = nn.Parameter(torch.randn(self.embedding_size))

        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

        # Pre-compute Pair Indices
        # Total fields = User (1) + Item (1) + Contexts (N)
        self.num_fields = 2 + len(self.context_labels)

        # Generate indices for all unique pairs (i, j) where i < j
        row_idx = []
        col_idx = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                row_idx.append(i)
                col_idx.append(j)

        # Register as buffers
        self.register_buffer("p_idx", torch.tensor(row_idx, dtype=torch.long))
        self.register_buffer("q_idx", torch.tensor(col_idx, dtype=torch.long))

        # Losses
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = EmbLoss()

        self.apply(self._init_weights)

    def _compute_afm_interaction(self, stacked_embeddings: Tensor) -> Tensor:
        """Computes the AFM interaction part."""
        # Pair-wise Interaction Layer
        # [batch_size, num_pairs, embedding_size]
        p = stacked_embeddings[:, self.p_idx]  # type: ignore[index]
        q = stacked_embeddings[:, self.q_idx]  # type: ignore[index]

        # Element-wise product
        pair_wise_inter = p * q

        # Apply Dropout on the interaction vectors
        pair_wise_inter = self.dropout_layer(pair_wise_inter)

        # Attention-based Pooling
        att_weights = self.attention_layer(
            pair_wise_inter
        )  # [batch_size, num_pairs, 1]

        # Weighted sum
        att_pooling = torch.sum(
            att_weights * pair_wise_inter, dim=1
        )  # [batch_size, embedding_size]

        # Final Projection
        afm_score = torch.sum(att_pooling * self.p, dim=1)  # [batch_size]

        return afm_score

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
        # Linear Part (First Order)
        linear_part = self.compute_first_order(user, item, contexts)

        # Prepare Embeddings
        # Order MUST be: User, Item, Contexts (to match p_idx/q_idx logic)
        embeddings_list = [self.user_embedding(user), self.item_embedding(item)]
        for idx, name in enumerate(self.context_labels):
            embeddings_list.append(self.context_embedding[name](contexts[:, idx]))

        stacked_embeddings = torch.stack(
            embeddings_list, dim=1
        )  # [batch_size, num_fields, embedding_size]

        # AFM Interaction Part
        afm_part = self._compute_afm_interaction(stacked_embeddings)

        return linear_part + afm_part

    def _compute_network_scores(
        self,
        u_emb: Tensor,
        i_emb: Tensor,
        ctx_emb_list: List[Tensor],
        batch_size: int,
        num_items: int,
    ) -> Tensor:
        """Compute scores of AFM interaction part efficiently using chunking."""
        total_rows = batch_size * num_items

        # Create memory efficient view
        u_view = u_emb.unsqueeze(1).expand(-1, num_items, -1).reshape(total_rows, -1)

        ctx_views = []
        for c in ctx_emb_list:
            ctx_views.append(
                c.unsqueeze(1).expand(-1, num_items, -1).reshape(total_rows, -1)
            )

        i_view = i_emb.reshape(total_rows, -1)

        # Pre-allocate tensor to memory
        all_scores = torch.empty(total_rows, device=self.device)

        # Loop on chunk size parameter
        for start in range(0, total_rows, self.eval_chunk_size):
            end = min(start + self.eval_chunk_size, total_rows)

            # Slice the views
            u_chunk = u_view[start:end]
            i_chunk = i_view[start:end]
            c_chunks = [c[start:end] for c in ctx_views]

            # Materialize ONLY the chunk
            # NOTE: This will actually use memory
            chunk_stack = torch.stack([u_chunk, i_chunk] + c_chunks, dim=1)

            # Compute AFM Interaction
            afm_s = self._compute_afm_interaction(chunk_stack)

            # Save in place
            all_scores[start:end] = afm_s

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
        """Prediction using the AFM model.

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
        for idx, name in enumerate(self.context_labels):
            fixed_linear += self.context_bias[name](contexts[:, idx]).squeeze(-1)

        # Embeddings Fixed
        u_emb = self.user_embedding(user_indices)  # [batch_size, embedding_size]
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

                items_block = torch.arange(start, end, device=self.device)

                # Item Embeddings and Bias
                item_emb_block = self.item_embedding(items_block)
                item_bias_block = self.item_bias(items_block).squeeze(-1)

                # Linear Part: [batch_size, 1] + [block_size] -> [batch_size, block_size]
                linear_pred = fixed_linear.unsqueeze(1) + item_bias_block.unsqueeze(0)

                # Expand Item to match batch size
                item_emb_expanded = item_emb_block.unsqueeze(0).expand(
                    batch_size, -1, -1
                )

                # Compute AFM scores efficiently
                afm_scores = self._compute_network_scores(
                    u_emb,
                    item_emb_expanded,
                    ctx_emb_list,
                    batch_size,
                    current_block_len,
                )

                preds_list.append(linear_pred + afm_scores)

            return torch.cat(preds_list, dim=1)

        else:
            # Case 'sampled': process given item_indices
            pad_seq = item_indices.size(1)

            # Item Embeddings
            item_emb = self.item_embedding(
                item_indices
            )  # [batch_size, pad_seq, embedding_size]
            item_bias = self.item_bias(item_indices).squeeze(
                -1
            )  # [batch_size, pad_seq]

            linear_pred = fixed_linear.unsqueeze(1) + item_bias

            # Compute AFM scores efficiently
            afm_scores = self._compute_network_scores(
                u_emb, item_emb, ctx_emb_list, batch_size, pad_seq
            )

            return linear_pred + afm_scores
