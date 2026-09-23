# pylint: disable = R0801, E1102
from typing import Any, List, Optional

import torch
from torch import Tensor, nn

from warprec.data.entities import Interactions, MultiModalFeatures, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.multimodal_recommender.multimodal_utils import (
    MultiModalRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="VBPR")
class VBPR(MultiModalRecommenderUtils, IterativeRecommender):
    """Implementation of VBPR algorithm from
        VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback (AAAI 2016)

    An item is two things at once: a factor learned from who interacted with it,
    and a projection of what it looks like. A user is two matching things, one
    reading each. Scoring is the dot product of the pair, which is the same as
    adding the collaborative agreement to the visual agreement, so the model
    falls back on the collaborative half for an item whose features are missing
    and leans on the visual half for an item almost nobody has touched.

    Nothing here fine-tunes the encoder that produced the features: only the
    projection out of them is learned, which is what makes the model cheap and
    what the paper does.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        multimodal (Optional[MultiModalFeatures]): The precomputed item features.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The width of both the collaborative and the
            projected feature factors.
        modalities (Optional[List[str]]): The modalities to read. Defaults to
            every configured one.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    embedding_size: int
    modalities: Optional[List[str]]
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        multimodal: Optional[MultiModalFeatures] = None,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, multimodal=multimodal, **kwargs)

        # The user reads both halves of the item, so it is twice as wide: a dot
        # product against the concatenation is the sum the paper writes out.
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size * 2)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        self.feature_projection = nn.Linear(
            sum(self.modality_dims), self.embedding_size, bias=False
        )

        # A per-item visual bias, which the paper carries beside the factors so
        # that a look people generally prefer shifts every user's score alike.
        self.visual_bias = nn.Linear(sum(self.modality_dims), 1, bias=False)
        self.item_bias = nn.Embedding(self.n_items + 1, 1, padding_idx=self.n_items)

        self.rec_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.apply(self._init_weights)

    def item_representation(self, item: Tensor) -> Tensor:
        """What an item is, collaboratively and visually.

        Args:
            item (Tensor): The item indices.

        Returns:
            Tensor: The item factors, the two halves concatenated.
        """
        features = self.joint_features()[item]
        return torch.cat(
            [self.item_embedding(item), self.feature_projection(features)], dim=-1
        )

    def item_offset(self, item: Tensor) -> Tensor:
        """How much an item is preferred before any user is considered.

        Args:
            item (Tensor): The item indices.

        Returns:
            Tensor: One bias per item.
        """
        features = self.joint_features()[item]
        return self.item_bias(item).squeeze(-1) + self.visual_bias(features).squeeze(-1)

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_contrastive_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        user, positive, negative = batch

        user_e = self.user_embedding(user)
        positive_e = self.item_representation(positive)
        negative_e = self.item_representation(negative)

        positive_score = (user_e * positive_e).sum(dim=1) + self.item_offset(positive)
        negative_score = (user_e * negative_e).sum(dim=1) + self.item_offset(negative)

        loss = self.rec_loss(positive_score, negative_score) + self.reg_weight * (
            self.reg_loss(user_e, positive_e, negative_e)
        )

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def forward(self, user: Tensor, item: Tensor) -> Tensor:
        """Score each given user against the item beside it.

        Args:
            user (Tensor): The user indices.
            item (Tensor): The item indices.

        Returns:
            Tensor: One score per pair.
        """
        pair = (self.user_embedding(user) * self.item_representation(item)).sum(dim=-1)
        return pair + self.item_offset(item)

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction from the collaborative and visual halves together.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        user_e = self.user_embedding(user_indices)

        if item_indices is None:
            catalogue = torch.arange(self.n_items, device=user_e.device)
            scores = user_e @ self.item_representation(catalogue).t()
            return scores + self.item_offset(catalogue).unsqueeze(0)

        scores = torch.einsum(
            "be,bse->bs", user_e, self.item_representation(item_indices)
        )
        return scores + self.item_offset(item_indices)
