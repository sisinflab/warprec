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


@model_registry.register(name="CKE")
class CKE(KnowledgeRecommenderUtils, IterativeRecommender):
    """Implementation of CKE algorithm from
        Collaborative Knowledge Base Embedding for Recommender Systems (KDD 2016)

    An item is represented by what people did with it and by what is known about
    it: a collaborative factor plus the embedding of the entity it stands for.
    The two are learned together, the first from the interactions and the second
    from the facts, so an item with few interactions still carries the structure
    its entity sits in.

    The facts are learned with TransR: a relation projects entities into a space
    of its own, where a true fact is the one whose head plus relation lands
    nearest its tail. Nothing is propagated over the graph, which is what makes
    this the simpler of the two knowledge-aware families.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        knowledge (Optional[KnowledgeGraph]): The facts about the items.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The width of the collaborative and entity factors.
        kg_embedding_size (int): The width of the space a relation projects into.
        reg_weight (float): The L2 regularization weight of the recommendation part.
        kg_reg_weight (float): The L2 regularization weight of the knowledge part.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    embedding_size: int
    kg_embedding_size: int
    reg_weight: float
    kg_reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

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
        self.item_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=self.n_items
        )

        # One row past the entities is the item the graph says nothing about.
        self.entity_embedding = nn.Embedding(
            self.n_entities + 1, self.embedding_size, padding_idx=self.n_entities
        )
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(
            self.n_relations, self.embedding_size * self.kg_embedding_size
        )

        self.rec_loss = BPRLoss()
        self.kg_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.apply(self._init_weights)

        self._triple_generator = torch.Generator()
        self._triple_generator.manual_seed(seed)

    def item_representation(self, item: Tensor) -> Tensor:
        """What an item is, collaboratively and factually.

        Args:
            item (Tensor): The item indices.

        Returns:
            Tensor: The item factors.
        """
        return self.item_embedding(item) + self.entity_embedding(self.entity_of(item))

    def _project(self, relation: Tensor, entity: Tensor) -> Tensor:
        """Put an entity into the space its relation defines.

        Args:
            relation (Tensor): The relation indices.
            entity (Tensor): The entity embeddings.

        Returns:
            Tensor: The projected embeddings, normalised.
        """
        projection = self.trans_w(relation).view(
            relation.size(0), self.embedding_size, self.kg_embedding_size
        )
        projected = torch.bmm(entity.unsqueeze(1), projection).squeeze(1)
        return F.normalize(projected, p=2, dim=1)

    def knowledge_loss(self, how_many: int) -> Tuple[Tensor, Tensor]:
        """How badly the facts are currently embedded.

        Args:
            how_many (int): How many facts to draw for this step.

        Returns:
            Tuple[Tensor, Tensor]: The translational loss and its regularizer.
        """
        heads, relations, tails, corrupted = self.sample_triples(
            how_many, self._triple_generator
        )

        head_e = self._project(relations, self.entity_embedding(heads))
        tail_e = self._project(relations, self.entity_embedding(tails))
        corrupted_e = self._project(relations, self.entity_embedding(corrupted))
        relation_e = F.normalize(self.relation_embedding(relations), p=2, dim=1)

        # A true fact should land nearer than a corrupted one, so the distances
        # are contrasted the way the scores of two items would be.
        true_distance = ((head_e + relation_e - tail_e) ** 2).sum(dim=1)
        false_distance = ((head_e + relation_e - corrupted_e) ** 2).sum(dim=1)

        loss = self.kg_loss(false_distance, true_distance)
        regularizer = self.reg_loss(head_e, relation_e, tail_e, corrupted_e)
        return loss, regularizer

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

        recommendation = self.rec_loss(
            (user_e * positive_e).sum(dim=1), (user_e * negative_e).sum(dim=1)
        )

        # The facts are learned beside the interactions, one batch of each, so
        # neither half of the model races ahead of the other.
        knowledge, knowledge_reg = self.knowledge_loss(user.size(0))

        loss = (
            recommendation
            + knowledge
            + self.reg_weight * self.reg_loss(user_e, positive_e, negative_e)
            + self.kg_reg_weight * knowledge_reg
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
        return (self.user_embedding(user) * self.item_representation(item)).sum(dim=-1)

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction from the collaborative and factual halves together.

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
            return user_e @ self.item_representation(catalogue).t()

        return torch.einsum(
            "be,bse->bs", user_e, self.item_representation(item_indices)
        )
