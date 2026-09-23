# pylint: disable = R0801, E1102
from typing import Any, List, Optional

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


class BiInteractionAggregator(nn.Module):
    """One propagation step: fold a node's neighbourhood back into the node.

    The two terms are what the bi-interaction aggregator is named for: the sum of
    a node and its neighbourhood, which carries what they have in common, and
    their element-wise product, which carries how they differ.

    Args:
        input_dim (int): The width coming in.
        output_dim (int): The width going out.
        dropout (float): The dropout applied to the result.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.sum_transform = nn.Linear(input_dim, output_dim)
        self.product_transform = nn.Linear(input_dim, output_dim)
        self.message_dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, adjacency: Tensor, embeddings: Tensor) -> Tensor:
        """Propagate once over the graph.

        Args:
            adjacency (Tensor): The sparse attention matrix.
            embeddings (Tensor): The current node embeddings.

        Returns:
            Tensor: The embeddings after one hop.
        """
        # The product against a sparse matrix is the aggregation: each row picks
        # up its neighbours, weighted by the attention on the edge.
        neighbourhood = torch.sparse.mm(adjacency, embeddings)

        shared = self.activation(self.sum_transform(embeddings + neighbourhood))
        distinct = self.activation(self.product_transform(embeddings * neighbourhood))
        return self.message_dropout(shared + distinct)


@model_registry.register(name="KGAT")
class KGAT(KnowledgeRecommenderUtils, IterativeRecommender):
    """Implementation of KGAT algorithm from
        KGAT: Knowledge Graph Attention Network for Recommendation (KDD 2019)

    Users, items and the entities behind them are put into one graph: an
    interaction is an edge like any other, so what is known about an item and who
    consumed it are propagated together. Each hop is weighted by an attention
    that asks how well a fact translates under its own relation, which lets the
    model prefer the facts that explain behaviour over the ones that do not.

    The graph is held sparsely and each hop is a product against it, which is
    what keeps a graph of millions of facts from materialising one message per
    fact.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        interactions (Optional[Interactions]): The training interactions, which
            become edges of the same graph.
        knowledge (Optional[KnowledgeGraph]): The facts about the items.
        seed (int): The seed to use for reproducibility.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        DATALOADER_TYPE: The type of dataloader used.
        embedding_size (int): The width of the node embeddings.
        kg_embedding_size (int): The width of the space a relation projects into.
        layers (List[int]): The width of each propagation hop.
        edge_heads (Tensor): The source of every edge of the collaborative graph.
        edge_tails (Tensor): The target of every edge.
        edge_relations (Tensor): The relation of every edge, interactions carrying
            one of their own.
        attention (Tensor): The sparse, row-normalised weight of every edge.
        dropout (float): The dropout applied after each hop.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.

    Raises:
        ValueError: If the training interactions were not provided.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    embedding_size: int
    kg_embedding_size: int
    layers: List[int]

    # Registered buffers, annotated so that they read as the tensors they are.
    edge_heads: Tensor
    edge_tails: Tensor
    edge_relations: Tensor
    attention: Tensor
    dropout: float
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        interactions: Optional[Interactions] = None,
        knowledge: Optional[KnowledgeGraph] = None,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(
            params,
            info,
            *args,
            interactions=interactions,
            knowledge=knowledge,
            seed=seed,
            **kwargs,
        )

        if interactions is None:
            raise ValueError(
                "KGAT puts the interactions into the graph beside the facts, so "
                "it needs them at construction."
            )

        # Users and entities share one table: in a collaborative knowledge graph
        # a user is a node like any other, and an item is the entity it stands for.
        self.n_nodes = self.n_users + self.n_entities + 1
        self.node_embedding = nn.Embedding(self.n_nodes, self.embedding_size)
        self.relation_embedding = nn.Embedding(
            self.n_relations + 1, self.kg_embedding_size
        )
        self.trans_w = nn.Embedding(
            self.n_relations + 1, self.embedding_size * self.kg_embedding_size
        )

        self.aggregators = nn.ModuleList()
        width = self.embedding_size
        for output in self.layers:
            self.aggregators.append(
                BiInteractionAggregator(width, output, self.dropout)
            )
            width = output

        self.rec_loss = BPRLoss()
        self.kg_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self._build_graph(interactions)
        self.apply(self._init_weights)

        self._triple_generator = torch.Generator()
        self._triple_generator.manual_seed(seed)

        # The attention is recomputed from the embeddings, so it starts uniform.
        self.register_buffer("attention", self._uniform_attention())

    def _build_graph(self, interactions: Interactions) -> None:
        """Lay the interactions and the facts out as one edge list.

        Args:
            interactions (Interactions): The training interactions.
        """
        matrix = interactions.get_sparse().tocoo()
        users = torch.as_tensor(matrix.row, dtype=torch.long)
        items = torch.as_tensor(matrix.col, dtype=torch.long)

        # A user is a node before the entities; an item is its entity.
        item_nodes = self.n_users + self.entity_of(items)
        entity_heads = self.n_users + self.triple_heads
        entity_tails = self.n_users + self.triple_tails

        # An interaction is a relation of its own, placed past the graph's own.
        interaction_relation = torch.full_like(users, self.n_relations)

        # Every edge is laid down both ways, so a hop can travel either
        # direction: a fact reaches the item it describes and the item reaches
        # back out to the fact.
        self.register_buffer(
            "edge_heads",
            torch.cat([users, item_nodes, entity_heads, entity_tails]),
        )
        self.register_buffer(
            "edge_tails",
            torch.cat([item_nodes, users, entity_tails, entity_heads]),
        )
        self.register_buffer(
            "edge_relations",
            torch.cat(
                [
                    interaction_relation,
                    interaction_relation,
                    self.triple_relations,
                    self.triple_relations,
                ]
            ),
        )

    def _uniform_attention(self) -> Tensor:
        """The graph before the attention has been learned.

        Returns:
            Tensor: The sparse adjacency, each edge weighted alike.
        """
        values = torch.ones(self.edge_heads.numel())
        indices = torch.stack([self.edge_heads, self.edge_tails])
        adjacency = torch.sparse_coo_tensor(
            indices, values, (self.n_nodes, self.n_nodes)
        ).coalesce()
        return torch.sparse.softmax(adjacency, dim=1)

    def _project(self, relation: Tensor, entity: Tensor) -> Tensor:
        """Put an entity into the space its relation defines.

        Args:
            relation (Tensor): The relation indices.
            entity (Tensor): The entity embeddings.

        Returns:
            Tensor: The projected embeddings.
        """
        projection = self.trans_w(relation).view(
            relation.size(0), self.embedding_size, self.kg_embedding_size
        )
        return torch.bmm(entity.unsqueeze(1), projection).squeeze(1)

    @torch.no_grad()
    def refresh_attention(self) -> None:
        """Recompute how much each edge should carry.

        An edge counts for as much as its fact translates well: the head, moved
        into the relation's space and shifted by it, against the tail. The result
        is normalised over each node's neighbourhood, which is the softmax the
        sparse layout gives directly.
        """
        heads = self.node_embedding(self.edge_heads)
        tails = self.node_embedding(self.edge_tails)
        relations = self.relation_embedding(self.edge_relations)

        projected_heads = self._project(self.edge_relations, heads)
        projected_tails = self._project(self.edge_relations, tails)
        scores = (projected_tails * torch.tanh(projected_heads + relations)).sum(dim=1)

        indices = torch.stack([self.edge_heads, self.edge_tails])
        adjacency = torch.sparse_coo_tensor(
            indices, scores, (self.n_nodes, self.n_nodes)
        ).coalesce()
        self.attention = torch.sparse.softmax(adjacency, dim=1)

    def propagate(self) -> Tensor:
        """Run every hop and keep what each one saw.

        Returns:
            Tensor: The node embeddings, each hop concatenated.
        """
        embeddings = self.node_embedding.weight

        # The raw embeddings enter the concatenation as they are: only what a hop
        # produced is normalised, which is what both the reference implementation
        # and RecBole do. Normalising the zeroth block too would rescale it
        # against the others and change the model.
        collected = [embeddings]

        for aggregator in self.aggregators:
            embeddings = aggregator(self.attention, embeddings)
            # The normalised copy is collected while the un-normalised one is
            # propagated onward, which is the order the reference uses.
            collected.append(F.normalize(embeddings, p=2, dim=1))

        return torch.cat(collected, dim=1)

    def on_train_epoch_start(self) -> None:
        """Refresh the attention before each pass over the data."""
        self.refresh_attention()

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

    def _knowledge_loss(self, how_many: int) -> Tensor:
        """How badly the facts currently translate.

        Args:
            how_many (int): How many facts to draw for this step.

        Returns:
            Tensor: The translational loss.
        """
        heads, relations, tails, corrupted = self.sample_triples(
            how_many, self._triple_generator
        )

        head_e = self._project(relations, self.node_embedding(self.n_users + heads))
        tail_e = self._project(relations, self.node_embedding(self.n_users + tails))
        corrupt_e = self._project(
            relations, self.node_embedding(self.n_users + corrupted)
        )
        relation_e = self.relation_embedding(relations)

        true_distance = ((head_e + relation_e - tail_e) ** 2).sum(dim=1)
        false_distance = ((head_e + relation_e - corrupt_e) ** 2).sum(dim=1)

        return self.kg_loss(false_distance, true_distance)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        user, positive, negative = batch

        nodes = self.propagate()
        user_e = nodes[user]
        positive_e = nodes[self.n_users + self.entity_of(positive)]
        negative_e = nodes[self.n_users + self.entity_of(negative)]

        recommendation = self.rec_loss(
            (user_e * positive_e).sum(dim=1), (user_e * negative_e).sum(dim=1)
        )
        knowledge = self._knowledge_loss(user.size(0))

        loss = (
            recommendation
            + knowledge
            + self.reg_weight * self.reg_loss(user_e, positive_e, negative_e)
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
        return (nodes[user] * nodes[self.n_users + self.entity_of(item)]).sum(dim=-1)

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction from the propagated embeddings.

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

        if item_indices is None:
            catalogue = torch.arange(self.n_items, device=user_e.device)
            return user_e @ nodes[self.n_users + self.entity_of(catalogue)].t()

        item_e = nodes[self.n_users + self.entity_of(item_indices)]
        return torch.einsum("be,bse->bs", user_e, item_e)
