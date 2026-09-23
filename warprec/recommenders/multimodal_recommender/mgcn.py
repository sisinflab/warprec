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
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="MGCN")
class MGCN(MultiModalRecommenderUtils, GraphRecommenderUtils, IterativeRecommender):
    """Implementation of MGCN algorithm from
        Multi-View Graph Convolutional Network for Multimedia Recommendation
        (MM 2023)

    A product photograph carries the product and also a watermark, a background
    and a house style; most of what a raw feature vector holds is not what makes
    the item worth recommending. MGCN's answer is to **purify** each modality
    against behaviour first: the features are gated by the item's collaborative
    embedding, so what survives is the part of the content that people actually
    responded to.

    What each modality then says is split into the part they agree on and the
    part where they differ, and how much a user cares about each difference is
    itself gated by their behaviour. A contrastive term keeps the content view
    and the behaviour view from drifting apart.

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
        knn_k (int): How many neighbours each item keeps in the feature graph.
        n_layers (int): How many hops to run over the item-item graph.
        n_ui_layers (int): How many hops to run over the user-item graph.
        modalities (Optional[List[str]]): The modalities to read. Defaults to
            every configured one.
        cl_weight (float): The weight of the content-behaviour agreement.
        temperature (float): The temperature of that agreement.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        item_item (Tensor): The frozen item-item feature graph.
        user_item (Tensor): The row-normalised user-item matrix.

    Raises:
        ValueError: If the training interactions were not provided.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    embedding_size: int
    knn_k: int
    n_layers: int
    n_ui_layers: int
    modalities: Optional[List[str]]
    cl_weight: float
    temperature: float
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    item_item: Tensor
    user_item: Tensor

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
                "MGCN propagates over the interactions, so it needs them at "
                "construction."
            )

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        self.projections = nn.ModuleList(
            nn.Linear(width, self.embedding_size) for width in self.modality_dims
        )
        # One gate per modality: what of the content survives is decided by what
        # the item's own collaborative embedding says is worth keeping.
        self.purifiers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size), nn.Sigmoid()
            )
            for _ in self.modality_names
        )
        self.preference_gates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size), nn.Tanh()
            )
            for _ in self.modality_names
        )
        self.common_query = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.Tanh(),
            nn.Linear(self.embedding_size, 1, bias=False),
        )

        self.bpr_loss = BPRLoss()
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
        self.register_buffer("item_item", self.feature_neighbour_graph(self.knn_k))
        self.register_buffer("user_item", self._build_user_item(interactions))

    def _build_user_item(self, interactions: Interactions) -> Tensor:
        """The user-item matrix, row-normalised.

        What a user thinks of a modality is the mean of what their items say in
        it, which is this matrix applied to the item side.

        Args:
            interactions (Interactions): The training interactions.

        Returns:
            Tensor: The sparse {user x (item + padding)} matrix.
        """
        matrix = interactions.get_sparse().tocoo()
        users = torch.as_tensor(matrix.row, dtype=torch.long)
        items = torch.as_tensor(matrix.col, dtype=torch.long)

        counts = torch.bincount(users, minlength=self.n_users).clamp(min=1).float()

        return torch.sparse_coo_tensor(
            torch.stack([users, items]),
            1.0 / counts[users],
            (self.n_users, self.n_items + 1),
        ).coalesce()

    def _modality_views(self) -> List[Tensor]:
        """Each modality, purified against behaviour and spread over the graph.

        Returns:
            List[Tensor]: One {node x embedding} view per modality.
        """
        views = []

        for projection, purifier, refined in zip(
            self.projections, self.purifiers, self.refined
        ):
            projected = projection(refined.weight)
            # The gate is the purification: the content is kept only where the
            # collaborative embedding says there is something to keep.
            purified = self.item_embedding.weight * purifier(projected)

            for _ in range(self.n_layers):
                purified = torch.sparse.mm(self.item_item, purified)

            side = torch.sparse.mm(self.user_item, purified)
            views.append(torch.cat([side, purified]))

        return views

    def forward(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build the behaviour view, the content view, and their sum.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The user embeddings, the item
                embeddings, the content view and the behaviour view.
        """
        ego = self.get_ego_embeddings(self.user_embedding, self.item_embedding)
        adjacency = self.adj
        if adjacency.device() != ego.device:
            adjacency = adjacency.to(ego.device)
            self.adj = adjacency

        collected = [ego]
        current = ego
        for _ in range(self.n_ui_layers):
            current = adjacency.matmul(current)
            collected.append(current)
        behaviour = torch.mean(torch.stack(collected, dim=0), dim=0)

        views = self._modality_views()

        # What the modalities agree on is weighted together; what is left over
        # is what each one says on its own.
        attention = torch.softmax(
            torch.cat([self.common_query(view) for view in views], dim=-1), dim=-1
        )
        common = sum(
            attention[:, index].unsqueeze(1) * view for index, view in enumerate(views)
        )

        # How much a user cares about a difference is gated by their behaviour.
        distinct = [
            gate(behaviour) * (view - common)
            for gate, view in zip(self.preference_gates, views)
        ]
        content = (sum(distinct) + common) / (len(views) + 1)

        combined = behaviour + content
        users, items = torch.split(combined, [self.n_users, self.n_items + 1])

        return users, items, content, behaviour

    def _agreement(self, first: Tensor, second: Tensor) -> Tensor:
        """How far the content view is from the behaviour view.

        Args:
            first (Tensor): One view of the batch's nodes.
            second (Tensor): The other.

        Returns:
            Tensor: The contrastive loss between them.
        """
        first, second = F.normalize(first, dim=1), F.normalize(second, dim=1)

        matched = torch.exp((first * second).sum(dim=-1) / self.temperature)
        against = torch.exp(first @ second.t() / self.temperature).sum(dim=1)

        return -torch.log(matched / against).mean()

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

        users, items, content, behaviour = self.forward()
        user_e, positive_e, negative_e = users[user], items[positive], items[negative]

        content_users, content_items = torch.split(
            content, [self.n_users, self.n_items + 1]
        )
        behaviour_users, behaviour_items = torch.split(
            behaviour, [self.n_users, self.n_items + 1]
        )

        loss = (
            self.bpr_loss(
                (user_e * positive_e).sum(dim=1), (user_e * negative_e).sum(dim=1)
            )
            + self.reg_weight * self.reg_loss(user_e, positive_e, negative_e)
            + self.cl_weight
            * (
                self._agreement(content_items[positive], behaviour_items[positive])
                + self._agreement(content_users[user], behaviour_users[user])
            )
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
        """Prediction from the behaviour and content views together.

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
        user_e = users[user_indices]

        if item_indices is None:
            return torch.einsum("be,ie->bi", user_e, items[: self.n_items])

        return torch.einsum("be,bse->bs", user_e, items[item_indices])
