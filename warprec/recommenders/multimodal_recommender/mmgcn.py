# pylint: disable = R0801, E1102
from typing import Any, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from warprec.data.entities import Interactions, MultiModalFeatures, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.multimodal_recommender.multimodal_utils import (
    MultiModalRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


class ModalityGraph(nn.Module):
    """One modality's own bipartite graph over users and items.

    Each modality is propagated separately, because what users share in how
    things look is not what they share in how things read. A user has a learned
    preference vector in each modality's space, the items enter it through their
    features, and the identity embedding is folded back in at every hop so the
    modality view stays tied to the collaborative one.

    Args:
        width (int): The width of the modality's features.
        hidden (int): The width the features are projected to.
        output (int): The width the representation comes out at.
        n_users (int): The number of users.
        n_layers (int): How many hops to run.
    """

    def __init__(
        self, width: int, hidden: int, output: int, n_users: int, n_layers: int
    ):
        # pylint: disable = too-many-arguments, too-many-positional-arguments
        super().__init__()

        self.preference = nn.Parameter(torch.empty(n_users, hidden))
        nn.init.xavier_normal_(self.preference)

        self.project = nn.Linear(width, hidden)

        # The reference unrolls three hops by name; the paper describes a loop,
        # so the depth is a parameter here and the reference is the n_layers=3
        # case of it.
        widths = [hidden] + [output] * n_layers
        self.aggregate = nn.ModuleList(
            nn.Linear(widths[hop], widths[hop + 1]) for hop in range(n_layers)
        )
        self.align = nn.ModuleList(
            nn.Linear(widths[hop], output) for hop in range(n_layers)
        )
        self.combine = nn.ModuleList(
            nn.Linear(widths[hop + 1], output) for hop in range(n_layers)
        )

    def forward(self, adjacency: Tensor, features: Tensor, identity: Tensor) -> Tensor:
        """Propagate this modality over the interaction graph.

        Args:
            adjacency (Tensor): The row-normalised bipartite adjacency.
            features (Tensor): The modality's item features.
            identity (Tensor): The shared identity embeddings of every node.

        Returns:
            Tensor: One representation per node.
        """
        nodes = F.normalize(torch.cat([self.preference, self.project(features)]))

        for hop, aggregate in enumerate(self.aggregate):
            neighbourhood = F.leaky_relu(aggregate(torch.sparse.mm(adjacency, nodes)))
            # The identity embedding enters at every hop, which is what keeps a
            # modality view from drifting away from the collaborative one.
            anchored = F.leaky_relu(self.align[hop](nodes)) + identity
            nodes = F.leaky_relu(self.combine[hop](neighbourhood) + anchored)

        return nodes


@model_registry.register(name="MMGCN")
class MMGCN(MultiModalRecommenderUtils, IterativeRecommender):
    """Implementation of MMGCN algorithm from
        MMGCN: Multi-modal Graph Convolution Network for Personalized
        Recommendation of Micro-video (MM 2019)

    The first multimodal graph model, and its argument is that the modalities
    should not be mixed before propagation. Two people who like the same look
    are not the same two people who like the same description, so each modality
    gets a bipartite graph of its own and is propagated on its own. Only the
    final representations are averaged, and a shared identity embedding folded
    in at each hop keeps the separate views anchored to one another.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        multimodal (Optional[MultiModalFeatures]): The precomputed item features.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The width of the representations.
        feature_size (int): The width each modality is projected to.
        n_layers (int): How many hops each modality graph runs.
        modalities (Optional[List[str]]): The modalities to read. Defaults to
            every configured one.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        adjacency (Tensor): The row-normalised bipartite adjacency.

    Raises:
        ValueError: If the training interactions were not provided.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    embedding_size: int
    feature_size: int
    n_layers: int
    modalities: Optional[List[str]]
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    adjacency: Tensor

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
                "MMGCN propagates over the interactions, so it needs them at "
                "construction."
            )

        self.identity = nn.Parameter(
            torch.empty(self.n_users + self.n_items + 1, self.embedding_size)
        )
        nn.init.xavier_normal_(self.identity)

        self.graphs = nn.ModuleList(
            ModalityGraph(
                width,
                self.feature_size,
                self.embedding_size,
                self.n_users,
                self.n_layers,
            )
            for width in self.modality_dims
        )

        self.bpr_loss = BPRLoss()
        self.register_buffer("adjacency", self._build_graph(interactions))

    def _build_graph(self, interactions: Interactions) -> Tensor:
        """Lay the interactions out as one row-normalised bipartite matrix.

        Args:
            interactions (Interactions): The training interactions.

        Returns:
            Tensor: The sparse adjacency, each row summing to one.
        """
        matrix = interactions.get_sparse().tocoo()
        users = torch.as_tensor(matrix.row, dtype=torch.long)
        items = torch.as_tensor(matrix.col, dtype=torch.long) + self.n_users

        rows = torch.cat([users, items])
        columns = torch.cat([items, users])
        side = self.n_users + self.n_items + 1

        # A mean over the neighbourhood is what the reference aggregates with,
        # which is a row-normalised product rather than a symmetric one.
        degree = torch.bincount(rows, minlength=side).clamp(min=1).float()

        return torch.sparse_coo_tensor(
            torch.stack([rows, columns]), 1.0 / degree[rows], (side, side)
        ).coalesce()

    def propagate(self) -> Tensor:
        """Average what every modality made of the graph.

        Returns:
            Tensor: One representation per node.
        """
        total = None
        for graph, features in zip(self.graphs, self.modality_tables()):
            # The padding item is a node like the others, so its (zero) features
            # travel with the rest.
            nodes = graph(self.adjacency, features, self.identity)
            total = nodes if total is None else total + nodes

        return total / len(self.graphs)

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

        nodes = self.propagate()
        user_e = nodes[user]
        positive_e = nodes[positive + self.n_users]
        negative_e = nodes[negative + self.n_users]

        regularizer = (
            self.identity[user] ** 2
            + self.identity[positive + self.n_users] ** 2
            + self.identity[negative + self.n_users] ** 2
        ).mean()

        loss = (
            self.bpr_loss(
                (user_e * positive_e).sum(dim=1), (user_e * negative_e).sum(dim=1)
            )
            + self.reg_weight * regularizer
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
        nodes = self.propagate()
        return (nodes[user] * nodes[item + self.n_users]).sum(dim=-1)

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction from the averaged per-modality representations.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        nodes = self.propagate()
        user_e = nodes[user_indices]
        items = nodes[self.n_users :]

        if item_indices is None:
            return torch.einsum("be,ie->bi", user_e, items[: self.n_items])

        return torch.einsum("be,bse->bs", user_e, items[item_indices])
