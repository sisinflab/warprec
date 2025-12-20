import torch
from torch import nn, Tensor
from typing import Any, Optional

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

    For further details, check the `paper <https://ieeexplore.ieee.org/document/5694074>`_.

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
        interactions: Interactions,
        info: dict,
        *args: Any,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, interactions, info, *args, seed=seed, **kwargs)

        # FM Layer (Interaction Part - Second Order)
        self.fm = FactorizationMachine(reduce_sum=True)

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
        """Forward pass of the FM model.

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
        interaction_part = self.fm(fm_input).squeeze(-1)

        return linear_part + interaction_part

    @torch.no_grad()
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

        # FM Fixed Accumulators
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
            # Case 'full': prediction on all items
            item_linear = self.item_bias.weight[:-1].squeeze(-1)  # [n_items]
            item_emb = self.item_embedding.weight[:-1, :]  # [n_items, embedding_size]

            final_linear = fixed_linear.unsqueeze(1) + item_linear.unsqueeze(
                0
            )  # [batch_size, n_items]

            # FM Interaction
            sum_v_fixed_exp = sum_v_fixed.unsqueeze(
                1
            )  # [batch_size, 1, embedding_size]
            sum_sq_v_fixed_exp = sum_sq_v_fixed.unsqueeze(1)
            item_emb_exp = item_emb.unsqueeze(0)  # [1, n_items, embedding_size]

            # Calculate (Sum V)^2 = (V_fixed + V_item)^2
            sum_all_sq = (sum_v_fixed_exp + item_emb_exp).pow(2)

            # Calculate Sum V^2 = (V_fixed^2 + V_item^2)
            sum_sq_all = sum_sq_v_fixed_exp + item_emb_exp.pow(2)

            # Final interaction term
            interaction = 0.5 * (sum_all_sq - sum_sq_all).sum(dim=2)

        else:
            # Case 'sampled': prediction on a sampled set of items
            if item_indices.dim() == 1:
                item_indices = item_indices.unsqueeze(1)  # [batch_size, 1]

            item_linear = self.item_bias(item_indices).squeeze(
                -1
            )  # [batch_size, pad_seq]
            item_emb = self.item_embedding(
                item_indices
            )  # [batch_size, pad_seq, embedding_size]

            final_linear = fixed_linear.unsqueeze(1) + item_linear

            sum_v_fixed_exp = sum_v_fixed.unsqueeze(
                1
            )  # [batch_size, 1, embedding_size]
            sum_sq_v_fixed_exp = sum_sq_v_fixed.unsqueeze(
                1
            )  # [batch_size, 1, embedding_size]

            sum_all_sq = (sum_v_fixed_exp + item_emb).pow(2)
            sum_sq_all = sum_sq_v_fixed_exp + item_emb.pow(2)

            interaction = 0.5 * (sum_all_sq - sum_sq_all).sum(dim=2)

        return final_linear + interaction
