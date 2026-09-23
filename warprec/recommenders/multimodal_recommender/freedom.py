# pylint: disable = R0801, E1102
from typing import Any, List, Optional, Tuple

import torch
from scipy.sparse import coo_matrix
from torch import Tensor, nn

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


@model_registry.register(name="FREEDOM")
class FREEDOM(MultiModalRecommenderUtils, GraphRecommenderUtils, IterativeRecommender):
    """Implementation of FREEDOM algorithm from
        A Tale of Two Graphs: Freezing and Denoising Graph Structures for
        Multimodal Recommendation (MM 2023)

    Two graphs carry the signal, and the model's argument is about what to do
    with each. The item-item graph, built by nearest neighbours over the raw
    features, is **frozen**: it is computed once before training and never
    learned, because learning it costs a great deal and buys nothing. The
    user-item graph is **denoised**: a share of its edges is dropped afresh every
    epoch, sampled against degree, so the model cannot come to depend on any one
    interaction.

    An item's representation is what its neighbourhood in the frozen graph says
    plus what propagation over the surviving interactions says. The features
    themselves also enter the loss directly, one contrast per modality, which is
    what keeps them aligned with the collaborative space. That contrast refines
    the feature vectors as well as their projection, as the reference
    implementation does; the frozen copy the graph was built from is untouched,
    so the graph stays the thing the paper froze.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        multimodal (Optional[MultiModalFeatures]): The precomputed item features.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The width of the collaborative embeddings.
        feature_size (int): The width the features are projected to.
        knn_k (int): How many neighbours each item keeps in the frozen graph.
        n_layers (int): How many hops to run over the frozen item-item graph.
        n_ui_layers (int): How many hops to run over the user-item graph.
        dropout (float): The share of interactions dropped each epoch.
        modalities (Optional[List[str]]): The modalities to read. Defaults to
            every configured one.
        modality_weights (Optional[List[float]]): How much each modality's
            item-item graph counts. Defaults to equal weight, applied after each
            graph is normalised on its own.
        reg_weight (float): The weight of the per-modality contrast.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        item_item (Tensor): The frozen item-item adjacency.
        edge_users (Tensor): The user end of each training interaction.
        edge_items (Tensor): The item end of each training interaction.
        edge_weights (Tensor): How likely each interaction is to survive a drop.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    embedding_size: int
    feature_size: int
    knn_k: int
    n_layers: int
    n_ui_layers: int
    dropout: float
    modalities: Optional[List[str]]
    modality_weights: Optional[List[float]]
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    # Registered buffers, annotated so that they read as the tensors they are
    # rather than as the union a buffer is typed with.
    item_item: Tensor
    edge_users: Tensor
    edge_items: Tensor
    edge_weights: Tensor

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        multimodal: Optional[MultiModalFeatures] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        # pylint: disable = too-many-arguments, too-many-positional-arguments
        super().__init__(
            params, info, *args, multimodal=multimodal, seed=seed, **kwargs
        )

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # One projection per modality: the widths differ, and a shared one would
        # force them into a common space before anything has aligned them.
        self.projections = nn.ModuleList(
            nn.Linear(width, self.feature_size) for width in self.modality_dims
        )

        # The reference implementation refines the feature vectors themselves
        # rather than only the projection out of them, so each modality gets a
        # trainable copy. The frozen originals stay where they are: the item-item
        # graph is built from them once and must not move afterwards.
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.apply(self._init_weights)

        # The refined copies are made after the initialiser has run, or it would
        # overwrite the features with random weights like any other embedding.
        # The copy also has to be its own storage: from_pretrained wraps the
        # tensor it is handed rather than copying it, so both the refinement and
        # the initialiser would otherwise write straight through into the frozen
        # features the item-item graph is built from.
        self.refined = nn.ModuleList(
            nn.Embedding.from_pretrained(self.modality(name).clone(), freeze=False)
            for name in self.modality_names
        )

        self._build_interaction_graph(interactions)
        self.register_buffer(
            "item_item",
            self.feature_neighbour_graph(self.knn_k, self.modality_weights),
        )

        self._edge_generator = torch.Generator()
        self._edge_generator.manual_seed(seed)

        # Before the first epoch begins the model still has to be able to score,
        # so it starts on the whole graph.
        self.masked_adj = self.adj

    def _build_interaction_graph(self, interactions: Interactions) -> None:
        """Keep the interactions in the two forms the epoch loop needs.

        Args:
            interactions (Interactions): The training interactions.
        """
        matrix = interactions.get_sparse().tocoo()
        self.adj = self.get_adj_mat(
            matrix, self.n_users, self.n_items + 1, normalize=True
        )

        rows = torch.as_tensor(matrix.row, dtype=torch.long)
        columns = torch.as_tensor(matrix.col, dtype=torch.long)
        self.register_buffer("edge_users", rows)
        self.register_buffer("edge_items", columns)

        # An edge is sampled against how well connected its ends already are, so
        # dropping thins the dense part of the graph rather than the sparse one.
        user_degree = torch.bincount(rows, minlength=self.n_users).float()
        item_degree = torch.bincount(columns, minlength=self.n_items + 1).float()
        weights = (user_degree[rows] * item_degree[columns]).pow(-0.5)
        self.register_buffer("edge_weights", torch.nan_to_num(weights, posinf=0.0))

    def on_train_epoch_start(self) -> None:
        """Draw the interactions this epoch is allowed to see."""
        self.masked_adj = self._drop_edges()

    def _drop_edges(self) -> Any:
        """Sample the surviving interactions and renormalise what is left.

        Returns:
            Any: The adjacency of the surviving interactions.
        """
        if self.dropout <= 0.0:
            return self.adj

        keep = int(self.edge_users.numel() * (1.0 - self.dropout))
        if keep <= 0:
            return self.adj

        chosen = torch.multinomial(
            self.edge_weights, keep, generator=self._edge_generator
        )

        matrix = coo_matrix(
            (
                torch.ones(keep).numpy(),
                (
                    self.edge_users[chosen].cpu().numpy(),
                    self.edge_items[chosen].cpu().numpy(),
                ),
            ),
            shape=(self.n_users, self.n_items + 1),
        )

        return self.get_adj_mat(
            matrix, self.n_users, self.n_items + 1, normalize=True
        ).to(self.item_embedding.weight.device)

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

    def forward(self) -> Tuple[Tensor, Tensor]:
        """Propagate over both graphs and put the two halves together.

        Returns:
            Tuple[Tensor, Tensor]: The user and item embeddings.
        """
        adjacency = self.masked_adj if self.training else self.adj

        # What the frozen graph says about an item, independent of any user.
        neighbourhood = self.item_embedding.weight
        if self.item_item.device != neighbourhood.device:
            self.item_item = self.item_item.to(neighbourhood.device)
        for _ in range(self.n_layers):
            neighbourhood = torch.sparse.mm(self.item_item, neighbourhood)

        ego = self.get_ego_embeddings(self.user_embedding, self.item_embedding)
        if adjacency.device() != ego.device:
            adjacency = adjacency.to(ego.device)

        collected = [ego]
        current = ego
        for _ in range(self.n_ui_layers):
            current = adjacency.matmul(current)
            collected.append(current)

        propagated = torch.mean(torch.stack(collected, dim=0), dim=0)
        users, items = torch.split(propagated, [self.n_users, self.n_items + 1])

        return users, items + neighbourhood

    def _modality_contrast(
        self, user_e: Tensor, positive: Tensor, negative: Tensor
    ) -> Tensor:
        """How well each modality alone separates the pair.

        Args:
            user_e (Tensor): The propagated user embeddings of the batch.
            positive (Tensor): The positive item indices.
            negative (Tensor): The negative item indices.

        Returns:
            Tensor: The summed per-modality contrast.
        """
        total = torch.zeros((), device=user_e.device)

        for refined, projection in zip(self.refined, self.projections):
            projected = projection(refined.weight)
            total = total + self.bpr_loss(
                (user_e * projected[positive]).sum(dim=1),
                (user_e * projected[negative]).sum(dim=1),
            )

        return total

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        user, positive, negative = batch

        users, items = self.forward()
        user_e = users[user]

        recommendation = self.bpr_loss(
            (user_e * items[positive]).sum(dim=1),
            (user_e * items[negative]).sum(dim=1),
        )

        # The projections are pulled towards the collaborative space by the same
        # contrast the recommendation is trained with, one modality at a time.
        loss = recommendation + self.reg_weight * self._modality_contrast(
            user_e, positive, negative
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
        """Prediction from both graphs together.

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
