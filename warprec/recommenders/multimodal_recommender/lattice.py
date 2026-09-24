# pylint: disable = R0801, E1102
from typing import Any, List, Optional, Tuple, cast

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


@model_registry.register(name="LATTICE")
class LATTICE(MultiModalRecommenderUtils, GraphRecommenderUtils, IterativeRecommender):
    """Implementation of LATTICE algorithm from
        Mining Latent Structures for Multimedia Recommendation (MM 2021)

    The structure between items is not given, it is learned. LATTICE keeps an
    item-item graph built by nearest neighbours over the **projected** features,
    and because the projection is trained the graph moves with it: the model
    discovers which items are alike rather than being told. The learned graph is
    mixed with the one built from the raw features, so it has somewhere to start
    from and cannot wander off.

    FREEDOM is the later argument that this learning is not worth its cost; both
    are here, and the difference between them is exactly ``lambda_coeff`` and
    whether the graph is rebuilt.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        multimodal (Optional[MultiModalFeatures]): The precomputed item features.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The width of the collaborative embeddings.
        feature_size (int): The width the features are projected to.
        knn_k (int): How many neighbours each item keeps.
        n_layers (int): How many hops to run over the item-item graph.
        n_ui_layers (int): How many hops to run over the user-item graph.
        lambda_coeff (float): How much of the raw-feature graph is kept.
        modalities (Optional[List[str]]): The modalities to read. Defaults to
            every configured one.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        frozen_item_graph (Tensor): The graph built from the raw features.

    Raises:
        ValueError: If the training interactions were not provided.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    embedding_size: int
    feature_size: int
    knn_k: int
    n_layers: int
    n_ui_layers: int
    lambda_coeff: float
    modalities: Optional[List[str]]
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    frozen_item_graph: Tensor

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
                "LATTICE propagates over the interactions, so it needs them at "
                "construction."
            )

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        self.projections = nn.ModuleList(
            nn.Linear(width, self.feature_size) for width in self.modality_dims
        )
        # How much each modality's graph counts is itself learned, which is what
        # lets the model decide that one modality is telling it more.
        self.modality_weight = nn.Parameter(
            torch.full((len(self.modality_names),), 1.0 / len(self.modality_names))
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
        self.register_buffer(
            "frozen_item_graph", self.feature_neighbour_graph(self.knn_k)
        )

        self._learned_graph: Optional[Tensor] = None

    def _learn_item_graph(self) -> Tensor:
        """Rebuild the item-item graph from the projections as they stand.

        Unlike the frozen graph this one carries a gradient, which is the whole
        of what LATTICE adds and the whole of what it costs.

        Returns:
            Tensor: The dense, normalised item-item adjacency.
        """
        share = torch.softmax(self.modality_weight, dim=0)
        combined = None

        for index, projection in enumerate(self.projections):
            vectors = cast(Tensor, self.refined[index].weight)
            features = F.normalize(projection(vectors[: self.n_items]), p=2, dim=1)
            similarity = features @ features.t()

            # Only the nearest neighbours survive; everything else is dropped,
            # which is what keeps the learned structure sparse in spirit.
            values, indices = torch.topk(similarity, self.knn_k, dim=-1)
            graph = torch.zeros_like(similarity).scatter_(1, indices, values)

            weighted = share[index] * graph
            combined = weighted if combined is None else combined + weighted

        degree = combined.sum(dim=1, keepdim=True) + 1e-7
        inverse = degree.pow(-0.5)

        return inverse * combined * inverse.t()

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Propagate over the learned item graph and the interaction graph.

        Returns:
            Tuple[Tensor, Tensor]: The user and item embeddings.
        """
        if self.training or self._learned_graph is None:
            self._learned_graph = self._learn_item_graph()

        learned = self._learned_graph
        frozen = self.frozen_item_graph
        if frozen.device != learned.device:
            frozen = frozen.to(learned.device)
            self.frozen_item_graph = frozen

        # The learned graph is anchored to the one the raw features give, so it
        # has somewhere to start and cannot drift away from the content.
        item_graph = (1 - self.lambda_coeff) * learned + self.lambda_coeff * (
            frozen.to_dense()[: self.n_items, : self.n_items]
        )

        neighbourhood = self.item_embedding.weight[: self.n_items]
        for _ in range(self.n_layers):
            neighbourhood = item_graph @ neighbourhood

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

        propagated = torch.mean(torch.stack(collected, dim=0), dim=0)
        users, items = torch.split(propagated, [self.n_users, self.n_items + 1])

        padded = F.pad(F.normalize(neighbourhood, p=2, dim=1), (0, 0, 0, 1))
        return users, items + padded

    def train(self, mode: bool = True) -> "LATTICE":
        """Drop the learned graph when the mode changes.

        Args:
            mode (bool): Whether the model is entering training mode.

        Returns:
            LATTICE: This model.
        """
        self._learned_graph = None
        return super().train(mode)

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

        users, items = self.forward()
        user_e, positive_e, negative_e = users[user], items[positive], items[negative]

        loss = self.bpr_loss(
            (user_e * positive_e).sum(dim=1), (user_e * negative_e).sum(dim=1)
        ) + self.reg_weight * self.reg_loss(user_e, positive_e, negative_e)

        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction from the learned structure and the interactions together.

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
