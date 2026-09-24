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

        Nothing here depends on the user, so a catalogue is walked once however
        many users are about to be scored against it.

        Args:
            entity (Tensor): The entity each row starts from.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: The entities reached at each hop
                and the relation each was reached by.
        """
        entities = [entity.unsqueeze(1)]
        relations = []

        for hop in range(self.n_iter):
            index = entities[hop].reshape(-1)
            entities.append(self.neighbour_entities[index].view(entity.size(0), -1))
            relations.append(self.neighbour_relations[index].view(entity.size(0), -1))

        return entities, relations

    def _aggregate(
        self,
        user_e: Tensor,
        entities: List[Tensor],
        relations: List[Tensor],
        paired: bool,
    ) -> Tensor:
        """Fold every hop back into one vector per item.

        Only the attention depends on the user; the walk itself does not. When a
        whole catalogue is being scored the walk is therefore gathered once and
        broadcast across the users, which is what keeps ranking from repeating
        the same gather for every one of them.

        Args:
            user_e (Tensor): The user embeddings, {user x embedding}.
            entities (List[Tensor]): The entities reached at each hop.
            relations (List[Tensor]): The relations they were reached by.
            paired (bool): Whether each walk belongs to the user beside it,
                rather than being shared across all of them.

        Returns:
            Tensor: The item embeddings, {user x item x embedding}.
        """
        width = self.embedding_size

        # Two alignments, one contraction. Paired, each walk belongs to the user
        # beside it, so the walk axis is the user axis and there is a single
        # item. Blocked, one walk per item is shared by every user, so the user
        # axis starts at one and broadcasts. Everything after this is the same.
        axis = 1 if paired else 0
        items = 1 if paired else entities[0].size(0)

        vectors = [self.entity_embedding(hop).unsqueeze(axis) for hop in entities]
        relation_vectors = [
            self.relation_embedding(hop).unsqueeze(axis) for hop in relations
        ]

        for step in range(self.n_iter):
            nearer = []
            for hop in range(self.n_iter - step):
                neighbourhood = vectors[hop + 1].view(
                    vectors[hop + 1].size(0), items, -1, self.neighbour_size, width
                )
                edges = relation_vectors[hop].view(
                    relation_vectors[hop].size(0),
                    items,
                    -1,
                    self.neighbour_size,
                    width,
                )

                # The agreement between the user and a relation is what decides
                # how much that edge carries, normalised across the neighbourhood.
                scores = (user_e.view(-1, 1, 1, 1, width) * edges).mean(dim=-1)
                weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
                summarised = (weights * neighbourhood).mean(dim=3)

                if self.aggregator == "sum":
                    output = vectors[hop] + summarised
                elif self.aggregator == "neighbour":
                    output = summarised
                else:
                    output = torch.cat(
                        [vectors[hop].expand_as(summarised), summarised], dim=-1
                    )

                output = self.transforms[step](output)

                # The last hop leaves the representation bounded, the earlier
                # ones keep it rectified, which is the order the paper uses.
                nearer.append(
                    torch.tanh(output)
                    if step == self.n_iter - 1
                    else torch.relu(output)
                )

            vectors = nearer

        return vectors[0].squeeze(2)

    def item_representation(self, user_e: Tensor, item: Tensor) -> Tensor:
        """What each item is, read through each user's eyes.

        Args:
            user_e (Tensor): The user embeddings, {user x embedding}.
            item (Tensor): The item indices to describe, one per column.

        Returns:
            Tensor: The item embeddings, {user x item x embedding}.
        """
        entities, relations = self._gather_hops(self.entity_of(item))
        return self._aggregate(user_e, entities, relations, paired=False)

    def _pair_representation(self, user_e: Tensor, item: Tensor) -> Tensor:
        """What one item is, for the user sitting beside it.

        Args:
            user_e (Tensor): The user embeddings of the batch.
            item (Tensor): One item per user.

        Returns:
            Tensor: One embedding per pair.
        """
        entities, relations = self._gather_hops(self.entity_of(item))
        return self._aggregate(user_e, entities, relations, paired=True).squeeze(1)

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
        positive_e = self._pair_representation(user_e, positive)
        negative_e = self._pair_representation(user_e, negative)

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
        return (user_e * self._pair_representation(user_e, item)).sum(dim=-1)

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

        # The walk over the catalogue is shared between the users of the batch,
        # so it is done once per block of items rather than once per pair.
        block = 512
        scores: List[Tensor] = []
        for start in range(0, items.size(1), block):
            chunk = items[0, start : start + block]
            walked = self.item_representation(user_e, chunk)
            scores.append(torch.einsum("be,bse->bs", user_e, walked))

        return torch.cat(scores, dim=1)
