# pylint: disable = R0801, E1102
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from warprec.data.entities import Interactions, MultiModalFeatures, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.collaborative_filtering_recommender.graph_based import (
    GraphRecommenderUtils,
)
from warprec.recommenders.multimodal_recommender.multimodal_utils import (
    MultiModalRecommenderUtils,
)
from warprec.recommenders.losses import EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="BM3")
class BM3(MultiModalRecommenderUtils, GraphRecommenderUtils, IterativeRecommender):
    """Implementation of BM3 algorithm from
        Bootstrap Latent Representations for Multi-modal Recommendation (WWW 2023)

    There are no negative samples anywhere in this model, which is its point.
    Sampling negatives is expensive and the items drawn are often not negative
    at all, merely unobserved. Instead the model makes two views of itself: an
    online one that is trained, and a target one that is the same representation
    with dropout applied and the gradient cut. Learning is pulling the online
    view towards the target, by cosine agreement, in three places at once — user
    against item, each modality against the item, and each modality against its
    own dropped-out self.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        multimodal (Optional[MultiModalFeatures]): The precomputed item features.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The width of the embeddings.
        n_layers (int): How many hops to run over the user-item graph.
        dropout (float): The dropout that makes the target view differ.
        modalities (Optional[List[str]]): The modalities to read. Defaults to
            every configured one.
        cl_weight (float): The weight of the per-modality agreement.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.

    Raises:
        ValueError: If the training interactions were not provided.
    """

    DATALOADER_TYPE = DataLoaderType.POS_DATALOADER

    embedding_size: int
    n_layers: int
    dropout: float
    modalities: Optional[List[str]]
    cl_weight: float
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        multimodal: Optional[MultiModalFeatures] = None,
        **kwargs: Any,
    ):
        # pylint: disable = too-many-arguments, too-many-positional-arguments
        super().__init__(params, info, *args, multimodal=multimodal, **kwargs)

        if interactions is None:
            raise ValueError(
                "BM3 propagates over the interactions, so it needs them at "
                "construction."
            )

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        self.projections = nn.ModuleList(
            nn.Linear(width, self.embedding_size) for width in self.modality_dims
        )
        # The predictor is what stops the two views collapsing onto one another.
        self.predictor = nn.Linear(self.embedding_size, self.embedding_size)

        self.reg_loss = EmbLoss()
        self.adj = self.get_adj_mat(
            interactions.get_sparse().tocoo(),
            self.n_users,
            self.n_items + 1,
            normalize=True,
        )

        self.apply(self._init_weights)

        self.refined = nn.ModuleList(
            nn.Embedding.from_pretrained(self.modality(name).clone(), freeze=False)
            for name in self.modality_names
        )

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Propagate over the interactions and keep the mean of every hop.

        Returns:
            Tuple[Tensor, Tensor]: The user and item embeddings.
        """
        ego = self.get_ego_embeddings(self.user_embedding, self.item_embedding)

        adjacency = self.adj
        if adjacency.device() != ego.device:
            adjacency = adjacency.to(ego.device)
            self.adj = adjacency

        collected = [ego]
        current = ego
        for _ in range(self.n_layers):
            current = adjacency.matmul(current)
            collected.append(current)

        propagated = torch.mean(torch.stack(collected, dim=0), dim=0)
        users, items = torch.split(propagated, [self.n_users, self.n_items + 1])

        return users, items + self.item_embedding.weight

    @staticmethod
    def _disagreement(online: Tensor, target: Tensor) -> Tensor:
        """How far apart the two views are, as a cosine distance.

        Args:
            online (Tensor): The view being trained.
            target (Tensor): The view being matched, already detached.

        Returns:
            Tensor: One minus their mean cosine agreement.
        """
        return (1 - F.cosine_similarity(online, target.detach(), dim=-1)).mean()

    def get_dataloader(
        self,
        interactions: Interactions,
        sessions: Sessions,
        **kwargs: Any,
    ):
        return interactions.get_positive_dataloader(
            batch_size=self.batch_size,
            **kwargs,
        )

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        user, item = batch[0], batch[1]

        users, items = self.forward()
        projected = [
            projection(refined.weight)
            for projection, refined in zip(self.projections, self.refined)
        ]

        # The target view is the same representation, dropped out and cut off
        # from the gradient. Nothing is sampled: this is what stands in for a
        # negative.
        with torch.no_grad():
            user_target = F.dropout(users.clone().detach(), self.dropout)
            item_target = F.dropout(items.clone().detach(), self.dropout)
            modality_targets = [
                F.dropout(block.clone().detach(), self.dropout) for block in projected
            ]

        user_online = self.predictor(users)[user]
        item_online = self.predictor(items)[item]

        loss = self._disagreement(user_online, item_target[item]) + self._disagreement(
            item_online, user_target[user]
        )

        agreement = torch.zeros((), device=loss.device)
        for block, target in zip(projected, modality_targets):
            predicted = self.predictor(block)[item]
            # Each modality is pulled towards the item it describes, and towards
            # its own dropped-out self.
            agreement = (
                agreement
                + self._disagreement(predicted, item_target[item])
                + self._disagreement(predicted, target[item])
            )

        loss = (
            loss
            + self.cl_weight * agreement
            + self.reg_weight * self.reg_loss(users[user], items[item])
        )

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction from the online view of both sides.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        users, items = self.propagate_embeddings()
        user_e = self.predictor(users)[user_indices]
        item_e = self.predictor(items)

        if item_indices is None:
            return torch.einsum("be,ie->bi", user_e, item_e[: self.n_items])

        return torch.einsum("be,bse->bs", user_e, item_e[item_indices])
