# pylint: disable = R0801, E1102
from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from warprec.data.entities import Interactions, KnowledgeGraph, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.knowledge_aware_recommender.knowledge_utils import (
    KnowledgeRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="KGIN")
class KGIN(KnowledgeRecommenderUtils, IterativeRecommender):
    """Implementation of KGIN algorithm from
        Learning Intents behind Interactions with Knowledge Graph for
        Recommendation (WWW 2021)

    An interaction is treated as the outcome of an **intent**, not as a bare
    link. The model keeps a small set of intents, each a learned mixture over
    the relations of the graph, and a user's taste is read as a distribution
    over them: someone whose intent is "same director" reads the graph through
    the directing edges, someone whose intent is "same genre" through the genre
    ones. The intents are pushed apart from each other, because two intents that
    say the same thing are one intent written twice.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        interactions (Interactions): The training interactions.
        *args (Any): Variable length argument list.
        knowledge (Optional[KnowledgeGraph]): The facts about the items.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The width of the embeddings.
        n_factors (int): How many intents to keep.
        n_hops (int): How many hops to propagate.
        node_dropout (float): The share of edges dropped each pass.
        mess_dropout (float): The dropout applied to each hop's output.
        independence (str): How intents are pushed apart, 'distance' or 'cosine'.
        ind_weight (float): The weight of that independence term.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        edge_heads (Tensor): The head of each fact, as a graph edge.
        edge_tails (Tensor): The tail of each fact, as a graph edge.
        edge_relations (Tensor): The relation each edge sits on.

    Raises:
        ValueError: If the training interactions were not provided.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    embedding_size: int
    n_factors: int
    n_hops: int
    node_dropout: float
    mess_dropout: float
    independence: str
    ind_weight: float
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    # Registered buffers, annotated so that they read as the tensors they are.
    edge_heads: Tensor
    edge_tails: Tensor
    edge_relations: Tensor

    def __init__(
        self,
        params: dict,
        info: dict,
        interactions: Interactions,
        *args: Any,
        knowledge: Optional[KnowledgeGraph] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        # pylint: disable = too-many-arguments, too-many-positional-arguments
        super().__init__(params, info, *args, knowledge=knowledge, seed=seed, **kwargs)

        if interactions is None:
            raise ValueError(
                "KGIN reads the interactions as one side of its graph, so it "
                "needs them at construction."
            )

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(
            self.n_entities + 1, self.embedding_size, padding_idx=self.n_entities
        )
        self.intent_embedding = nn.Embedding(self.n_factors, self.embedding_size)

        # A relation is a gate on the message that travels along it, and an
        # intent is a mixture over those gates.
        self.relation_weight = nn.Parameter(
            torch.empty(self.n_relations, self.embedding_size)
        )
        self.intent_over_relations = nn.Parameter(
            torch.empty(self.n_factors, self.n_relations)
        )
        nn.init.xavier_uniform_(self.relation_weight)
        nn.init.xavier_uniform_(self.intent_over_relations)

        self.dropout = nn.Dropout(p=self.mess_dropout)
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.apply(self._init_weights)

        self._build_graph(interactions)
        self._edge_generator = torch.Generator()
        self._edge_generator.manual_seed(seed)

    def _build_graph(self, interactions: Interactions) -> None:
        """Keep the facts as an edge list and the interactions as a matrix.

        Args:
            interactions (Interactions): The training interactions.
        """
        heads, relations, tails = self._graph.get_triples()

        # Every fact is laid down both ways, so a message can travel from an
        # item out to what is known about it and back again.
        self.register_buffer("edge_heads", torch.cat([heads, tails]))
        self.register_buffer("edge_tails", torch.cat([tails, heads]))
        self.register_buffer("edge_relations", torch.cat([relations, relations]))

        matrix = interactions.get_sparse().tocoo()
        users = torch.as_tensor(matrix.row, dtype=torch.long)
        items = self.entity_of(torch.as_tensor(matrix.col, dtype=torch.long))

        # The user side is a plain mean over the entities a user touched, so it
        # is held as a normalised sparse matrix rather than an edge list.
        counts = torch.bincount(users, minlength=self.n_users).clamp(min=1).float()
        self.interactions = torch.sparse_coo_tensor(
            torch.stack([users, items]),
            (1.0 / counts[users]),
            (self.n_users, self.n_entities + 1),
        ).coalesce()

    def _aggregate(self, entity_e: Tensor, user_e: Tensor) -> Tuple[Tensor, Tensor]:
        """Run one hop over the facts and over the interactions.

        Args:
            entity_e (Tensor): The current entity embeddings.
            user_e (Tensor): The current user embeddings.

        Returns:
            Tuple[Tensor, Tensor]: The aggregated entity and user embeddings.
        """
        heads, tails, relations = self.edge_heads, self.edge_tails, self.edge_relations

        if self.training and self.node_dropout > 0.0:
            keep = torch.rand(heads.numel(), generator=self._edge_generator)
            keep = keep >= self.node_dropout
            heads, tails, relations = heads[keep], tails[keep], relations[keep]

        # A message is gated by the relation it travels along, which is an
        # elementwise product per edge: unlike a plain propagation this cannot
        # be folded into one sparse product, so the messages are materialised
        # and scattered back.
        messages = entity_e[tails] * self.relation_weight[relations]
        gathered = torch.zeros_like(entity_e).index_add_(0, heads, messages)
        counts = (
            torch.zeros(entity_e.size(0), device=entity_e.device)
            .index_add_(0, heads, torch.ones_like(heads, dtype=torch.float))
            .clamp(min=1.0)
        )
        entity_agg = gathered / counts.unsqueeze(1)

        # What a user picks up is what their intents say the relations are worth.
        neighbourhood = torch.sparse.mm(self.interactions, entity_e)
        attention = torch.softmax(user_e @ self.intent_embedding.weight.t(), dim=1)
        intent_gates = torch.softmax(self.intent_over_relations, dim=-1) @ (
            self.relation_weight
        )
        user_agg = (
            neighbourhood
            * (attention.unsqueeze(-1) * intent_gates.unsqueeze(0)).sum(dim=1)
            + neighbourhood
        )

        return entity_agg, user_agg

    def independence_loss(self) -> Tensor:
        """How much the intents overlap with one another.

        Two intents that describe the same relations are one intent written
        twice, so the model is pushed to keep them apart.

        Returns:
            Tensor: The overlap, summed over every pair of intents.
        """
        gates = torch.softmax(self.intent_over_relations, dim=-1) @ self.relation_weight

        if self.independence == "cosine":
            normalised = F.normalize(gates, p=2, dim=1)
            overlap = normalised @ normalised.t()
            # Only the distinct pairs count; an intent always agrees with itself.
            return torch.triu(overlap, diagonal=1).abs().sum()

        total = torch.zeros((), device=gates.device)
        for first in range(self.n_factors - 1):
            for second in range(first + 1, self.n_factors):
                total = total + self._distance_correlation(gates[first], gates[second])
        return total

    @staticmethod
    def _distance_correlation(first: Tensor, second: Tensor) -> Tensor:
        """The distance correlation between two intent gates.

        Args:
            first (Tensor): One intent's gate over the relations.
            second (Tensor): The other's.

        Returns:
            Tensor: Their distance correlation, zero when independent.
        """

        def centred(vector: Tensor) -> Tensor:
            column = vector.unsqueeze(-1)
            squared = column**2
            distances = torch.sqrt(
                torch.clamp(squared - 2 * (column @ column.t()) + squared.t(), min=0.0)
                + 1e-8
            )
            return (
                distances
                - distances.mean(dim=0, keepdim=True)
                - distances.mean(dim=1, keepdim=True)
                + distances.mean()
            )

        left, right = centred(first), centred(second)
        channel = first.size(0) ** 2

        covariance = torch.sqrt(
            torch.clamp((left * right).sum() / channel, min=0.0) + 1e-8
        )
        left_variance = torch.sqrt(
            torch.clamp((left * left).sum() / channel, min=0.0) + 1e-8
        )
        right_variance = torch.sqrt(
            torch.clamp((right * right).sum() / channel, min=0.0) + 1e-8
        )

        return covariance / torch.sqrt(left_variance * right_variance + 1e-8)

    def propagate(self) -> Tuple[Tensor, Tensor]:
        """Run every hop and sum what each one saw.

        Returns:
            Tuple[Tensor, Tensor]: The user and entity embeddings.
        """
        entity_e = self.entity_embedding.weight
        user_e = self.user_embedding.weight

        entity_result, user_result = entity_e, user_e

        for _ in range(self.n_hops):
            entity_e, user_e = self._aggregate(entity_e, user_e)

            if self.training:
                entity_e = self.dropout(entity_e)
                user_e = self.dropout(user_e)

            entity_e = F.normalize(entity_e)
            user_e = F.normalize(user_e)

            entity_result = entity_result + entity_e
            user_result = user_result + user_e

        return user_result, entity_result

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

        users, entities = self.propagate()
        user_e = users[user]
        positive_e = entities[self.entity_of(positive)]
        negative_e = entities[self.entity_of(negative)]

        loss = (
            self.bpr_loss(
                (user_e * positive_e).sum(dim=1), (user_e * negative_e).sum(dim=1)
            )
            + self.reg_weight * self.reg_loss(user_e, positive_e, negative_e)
            + self.ind_weight * self.independence_loss()
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
        users, entities = self.propagate()
        return (users[user] * entities[self.entity_of(item)]).sum(dim=-1)

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction from the propagated, intent-weighted embeddings.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        users, entities = self.propagate()
        user_e = users[user_indices]

        if item_indices is None:
            catalogue = torch.arange(self.n_items, device=user_e.device)
            return user_e @ entities[self.entity_of(catalogue)].t()

        return torch.einsum(
            "be,bse->bs", user_e, entities[self.entity_of(item_indices)]
        )
