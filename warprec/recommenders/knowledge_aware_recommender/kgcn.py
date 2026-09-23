# pylint: disable = R0801, E1102
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor, nn

from warprec.data.entities import Interactions, KnowledgeGraph, Sessions
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.recommenders.knowledge_aware_recommender.knowledge_utils import (
    KnowledgeRecommenderUtils,
)
from warprec.recommenders.losses import BPRLoss, EmbLoss
from warprec.utils.enums import DataLoaderType
from warprec.utils.registry import model_registry


@model_registry.register(name="KGCN")
class KGCN(KnowledgeRecommenderUtils, IterativeRecommender):
    """Implementation of KGCN algorithm from
        Knowledge Graph Convolutional Networks for Recommender Systems (WWW 2019)

    An item is described by the entities around it, but not every relation
    matters equally to every user: someone who picks films by director should
    have the "directed by" edges count for more than the "genre" ones. KGCN
    makes that explicit. The weight of an edge is the agreement between the user
    and the relation it was reached by, normalised over the neighbourhood, so
    each user effectively reads a graph of their own.

    The neighbourhood is sampled to a fixed size once, before training, which is
    what lets the whole gather be a rectangle rather than a ragged walk.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        knowledge (Optional[KnowledgeGraph]): The facts about the items.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The width of the entity and relation embeddings.
        neighbour_size (int): How many neighbours each entity is given.
        n_iter (int): How many hops out from an item to read.
        aggregator (str): How a node is combined with its neighbourhood.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        neighbour_entities (Tensor): The sampled neighbours of every entity.
        neighbour_relations (Tensor): How each of those neighbours was reached.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    embedding_size: int
    neighbour_size: int
    n_iter: int
    aggregator: str
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    # Registered buffers, annotated so that they read as the tensors they are.
    neighbour_entities: Tensor
    neighbour_relations: Tensor

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        knowledge: Optional[KnowledgeGraph] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, knowledge=knowledge, seed=seed, **kwargs)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        # One row past the entities is the item the graph says nothing about.
        self.entity_embedding = nn.Embedding(
            self.n_entities + 1, self.embedding_size, padding_idx=self.n_entities
        )
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)

        width = self.embedding_size * (2 if self.aggregator == "concat" else 1)
        self.transforms = nn.ModuleList(
            nn.Linear(width, self.embedding_size) for _ in range(self.n_iter)
        )

        self.rec_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.apply(self._init_weights)

        sampler = torch.Generator()
        sampler.manual_seed(seed)
        entities, relations = knowledge.sample_neighbours(self.neighbour_size, sampler)

        # The padding entity is its own neighbourhood, so an item the graph is
        # silent about gathers zeros rather than somebody else's facts.
        padding_entity = torch.full((1, self.neighbour_size), self.n_entities)
        padding_relation = torch.zeros((1, self.neighbour_size), dtype=torch.long)
        self.register_buffer(
            "neighbour_entities", torch.cat([entities, padding_entity])
        )
        self.register_buffer(
            "neighbour_relations", torch.cat([relations, padding_relation])
        )

    def _gather_hops(self, entity: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """Walk out from the given entities, one hop at a time.

        Args:
            entity (Tensor): The entity each row starts from.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: The entities reached at each hop
                and the relation each was reached by.
        """
        entities = [entity.unsqueeze(1)]
        relations = []

        for hop in range(self.n_iter):
            index = entities[hop].flatten()
            entities.append(self.neighbour_entities[index].view(entity.size(0), -1))
            relations.append(self.neighbour_relations[index].view(entity.size(0), -1))

        return entities, relations

    def _mix(self, neighbours: Tensor, relations: Tensor, user_e: Tensor) -> Tensor:
        """Weigh a neighbourhood by what the user cares about.

        Args:
            neighbours (Tensor): The neighbour embeddings.
            relations (Tensor): The embeddings of the relations they sit on.
            user_e (Tensor): The user embeddings of the batch.

        Returns:
            Tensor: One vector per node, its neighbourhood summarised.
        """
        # The agreement between the user and a relation is what decides how much
        # that edge carries, normalised across the neighbourhood.
        scores = (user_e.view(-1, 1, 1, self.embedding_size) * relations).mean(dim=-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)

        return (weights * neighbours).mean(dim=2)

    def _aggregate(
        self, user_e: Tensor, entities: List[Tensor], relations: List[Tensor]
    ) -> Tensor:
        """Fold every hop back into one vector per item.

        Args:
            user_e (Tensor): The user embeddings of the batch.
            entities (List[Tensor]): The entities reached at each hop.
            relations (List[Tensor]): The relations they were reached by.

        Returns:
            Tensor: One embedding per item of the batch.
        """
        batch = user_e.size(0)
        vectors = [self.entity_embedding(hop) for hop in entities]
        relation_vectors = [self.relation_embedding(hop) for hop in relations]

        for step in range(self.n_iter):
            nearer = []
            for hop in range(self.n_iter - step):
                shape = (batch, -1, self.neighbour_size, self.embedding_size)
                summarised = self._mix(
                    vectors[hop + 1].view(shape),
                    relation_vectors[hop].view(shape),
                    user_e,
                )

                if self.aggregator == "sum":
                    output = vectors[hop] + summarised
                elif self.aggregator == "neighbour":
                    output = summarised
                else:
                    output = torch.cat([vectors[hop], summarised], dim=-1)

                output = self.transforms[step](output.view(batch, -1, output.size(-1)))

                # The last hop leaves the representation bounded, the earlier
                # ones keep it rectified, which is the order the paper uses.
                nearer.append(
                    torch.tanh(output)
                    if step == self.n_iter - 1
                    else torch.relu(output)
                )

            vectors = nearer

        return vectors[0].view(batch, self.embedding_size)

    def item_representation(self, user_e: Tensor, item: Tensor) -> Tensor:
        """What an item is, read through one user's eyes.

        Args:
            user_e (Tensor): The user embeddings of the batch.
            item (Tensor): The item indices.

        Returns:
            Tensor: One embedding per item.
        """
        entities, relations = self._gather_hops(self.entity_of(item))
        return self._aggregate(user_e, entities, relations)

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
        positive_e = self.item_representation(user_e, positive)
        negative_e = self.item_representation(user_e, negative)

        loss = self.rec_loss(
            (user_e * positive_e).sum(dim=1), (user_e * negative_e).sum(dim=1)
        ) + self.reg_weight * self.reg_loss(user_e, positive_e, negative_e)

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
        user_e = self.user_embedding(user)
        return (user_e * self.item_representation(user_e, item)).sum(dim=-1)

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction over a user-specific reading of the graph.

        Every item has to be walked once per user, because the weights the walk
        uses are the user's own, so the catalogue is scored in chunks rather
        than as one matrix.

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
            items = torch.arange(self.n_items, device=user_e.device)
            items = items.unsqueeze(0).expand(user_indices.size(0), -1)
        else:
            items = item_indices

        flat_users = (
            user_e.unsqueeze(1)
            .expand(-1, items.size(1), -1)
            .reshape(-1, self.embedding_size)
        )
        scored = self.item_representation(flat_users, items.reshape(-1))

        return (flat_users * scored).sum(dim=-1).view(items.size(0), items.size(1))
