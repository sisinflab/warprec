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


@model_registry.register(name="RippleNet")
class RippleNet(KnowledgeRecommenderUtils, IterativeRecommender):
    """Implementation of RippleNet algorithm from
        RippleNet: Propagating User Preferences on the Knowledge Graph for
        Recommender Systems (CIKM 2018)

    A user is not represented by a learned vector at all. What stands for them
    is the set of facts reachable from the things they have already interacted
    with: the items themselves, then the entities one fact away, then two, like
    ripples spreading out from where a stone landed. Each ring is addressed
    against the candidate item, the facts that speak to it most are weighted
    highest, and the rings are summed into the vector the item is scored against.

    The rings are sampled to a fixed size, because a user's reachable set grows
    very fast and the model only ever needs a sample of it.

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
        embedding_size (int): The width of the entity embeddings.
        n_hop (int): How many rings to spread out.
        n_memory (int): How many facts each ring holds.
        kg_weight (float): The weight of the fact-plausibility term.
        reg_weight (float): The L2 regularization weight.
        batch_size (int): The batch size used for training.
        epochs (int): The number of epochs.
        learning_rate (float): The learning rate value.
        ripple_heads (Tensor): The head of each remembered fact, per user and ring.
        ripple_relations (Tensor): The relation of each remembered fact.
        ripple_tails (Tensor): The tail of each remembered fact.

    Raises:
        ValueError: If the training interactions were not provided.
    """

    DATALOADER_TYPE = DataLoaderType.POS_NEG_LOADER

    embedding_size: int
    n_hop: int
    n_memory: int
    kg_weight: float
    reg_weight: float
    batch_size: int
    epochs: int
    learning_rate: float

    # Registered buffers, annotated so that they read as the tensors they are.
    ripple_heads: Tensor
    ripple_relations: Tensor
    ripple_tails: Tensor

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
                "RippleNet spreads out from what a user has already interacted "
                "with, so it needs the interactions at construction."
            )

        self.entity_embedding = nn.Embedding(
            self.n_entities + 1, self.embedding_size, padding_idx=self.n_entities
        )
        # A relation is a matrix here, not a vector: a ring is addressed by
        # moving the head through the relation and meeting the item there.
        self.relation_embedding = nn.Embedding(
            self.n_relations, self.embedding_size * self.embedding_size
        )
        self.transform = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        self.rec_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.apply(self._init_weights)

        generator = torch.Generator()
        generator.manual_seed(seed)
        heads, relations, tails = self._build_ripples(interactions, generator)
        self.register_buffer("ripple_heads", heads)
        self.register_buffer("ripple_relations", relations)
        self.register_buffer("ripple_tails", tails)

    def _build_ripples(
        self, interactions: Interactions, generator: torch.Generator
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Spread out from every user's history, one ring at a time.

        Args:
            interactions (Interactions): The training interactions.
            generator (torch.Generator): The stream the sampling draws from.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The heads, relations and tails of each
                ring, shaped {hop x user x memory}.
        """
        offsets, neighbours, relations = self._graph.neighbour_index()
        matrix = interactions.get_sparse().tocsr()

        heads = torch.full((self.n_hop, self.n_users, self.n_memory), self.n_entities)
        ring_relations = torch.zeros(
            (self.n_hop, self.n_users, self.n_memory), dtype=torch.long
        )
        tails = torch.full((self.n_hop, self.n_users, self.n_memory), self.n_entities)

        for user in range(self.n_users):
            # The first ring starts at the entities the user's own items stand
            # for; later rings start where the previous one landed.
            seeds = self.entity_of(
                torch.as_tensor(
                    matrix.indices[matrix.indptr[user] : matrix.indptr[user + 1]]
                )
            )
            seeds = seeds[seeds < self.n_entities]

            for hop in range(self.n_hop):
                drawn = self._draw_ring(
                    seeds, offsets, neighbours, relations, generator
                )
                if drawn is None:
                    # Nothing reachable: the ring stays padding, and so does
                    # every ring after it.
                    break

                heads[hop, user], ring_relations[hop, user], tails[hop, user] = drawn
                seeds = tails[hop, user]

        return heads, ring_relations, tails

    def _draw_ring(
        self,
        seeds: Tensor,
        offsets: Tensor,
        neighbours: Tensor,
        relations: Tensor,
        generator: torch.Generator,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
        """Sample one ring of facts reachable from the given entities.

        Args:
            seeds (Tensor): The entities this ring spreads out from.
            offsets (Tensor): The neighbour-list offsets of the graph.
            neighbours (Tensor): The neighbour entities of the graph.
            relations (Tensor): The relation each neighbour sits on.
            generator (torch.Generator): The stream the sampling draws from.

        Returns:
            Optional[Tuple[Tensor, Tensor, Tensor]]: The sampled heads, relations
                and tails, or None when nothing is reachable.
        """
        # pylint: disable = too-many-arguments, too-many-positional-arguments
        usable = seeds[
            (seeds < self.n_entities) & (offsets[seeds + 1] > offsets[seeds])
        ]
        if usable.numel() == 0:
            return None

        counts = offsets[usable + 1] - offsets[usable]

        # Draw the facts uniformly over the whole ring rather than per seed, so
        # a densely connected seed contributes proportionally more of it.
        total = int(counts.sum())
        chosen = torch.randint(0, total, (self.n_memory,), generator=generator)

        boundaries = torch.cumsum(counts, dim=0)
        which_seed = torch.searchsorted(boundaries, chosen, right=True)
        within = chosen - (boundaries[which_seed] - counts[which_seed])

        positions = offsets[usable[which_seed]] + within

        return usable[which_seed], relations[positions], neighbours[positions]

    def _rings_of(self, user: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """Gather each ring once, already moved through its relations.

        Moving a head through its relation does not depend on what is being
        scored, so it is done here rather than once per candidate item. That
        product is by far the most expensive thing the model does, because a
        relation is a matrix.

        Args:
            user (Tensor): The users whose rings to gather.

        Returns:
            List[Tuple[Tensor, Tensor]]: Per hop, the moved heads and the tails.
        """
        rings = []

        for hop in range(self.n_hop):
            heads = self.entity_embedding(self.ripple_heads[hop][user])
            tails = self.entity_embedding(self.ripple_tails[hop][user])
            matrices = self.relation_embedding(self.ripple_relations[hop][user]).view(
                -1, self.n_memory, self.embedding_size, self.embedding_size
            )
            rings.append((torch.einsum("bmij,bmj->bmi", matrices, heads), tails))

        return rings

    def _address(self, rings: List[Tuple[Tensor, Tensor]], item_e: Tensor) -> Tensor:
        """Read every ring against the items being scored.

        Args:
            rings (List[Tuple[Tensor, Tensor]]): The gathered rings, per hop.
            item_e (Tensor): The candidate item embeddings, {batch x item x dim}.

        Returns:
            Tensor: The user representation each item is scored against.
        """
        response = torch.zeros_like(item_e)
        current = item_e

        for moved, tails in rings:
            # The facts that land nearest the item are the ones that count. Every
            # candidate is read against the same rings, so this is one product
            # over the whole block rather than one per item.
            scores = torch.einsum("bmi,bsi->bsm", moved, current)
            weights = torch.softmax(scores, dim=-1)

            ring = torch.einsum("bsm,bmi->bsi", weights, tails)
            response = response + ring

            # The item is nudged towards what the ring said before the next one
            # is read, which is what makes the rings sequential rather than
            # independent.
            current = self.transform(current + ring)

        return response

    def _plausibility(self, user: Tensor) -> Tensor:
        """How well the remembered facts hold together as facts.

        Args:
            user (Tensor): The users whose rings to score.

        Returns:
            Tensor: The summed plausibility of their facts.
        """
        total = torch.zeros((), device=user.device)

        for hop in range(self.n_hop):
            heads = self.entity_embedding(self.ripple_heads[hop][user])
            tails = self.entity_embedding(self.ripple_tails[hop][user])
            matrices = self.relation_embedding(self.ripple_relations[hop][user]).view(
                -1, self.n_memory, self.embedding_size, self.embedding_size
            )

            moved = torch.einsum("bmi,bmij->bmj", heads, matrices)
            total = total + torch.sigmoid((moved * tails).sum(dim=-1)).mean()

        return total

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

        rings = self._rings_of(user)
        positive_e = self.entity_embedding(self.entity_of(positive)).unsqueeze(1)
        negative_e = self.entity_embedding(self.entity_of(negative)).unsqueeze(1)

        positive_score = (positive_e * self._address(rings, positive_e)).sum(dim=-1)
        negative_score = (negative_e * self._address(rings, negative_e)).sum(dim=-1)

        # A fact that does not hold as a fact should not be carrying preference,
        # so the rings are pushed towards being plausible as well as useful.
        loss = (
            self.rec_loss(positive_score, negative_score)
            - self.kg_weight * self._plausibility(user)
            + self.reg_weight * self.reg_loss(positive_e, negative_e)
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
        item_e = self.entity_embedding(self.entity_of(item)).unsqueeze(1)
        scored = (item_e * self._address(self._rings_of(user), item_e)).sum(dim=-1)
        return scored.squeeze(1)

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Prediction by reading each user's rings against every candidate.

        The rings are addressed against the item, so there is no user vector to
        multiply a catalogue by: each pair is read on its own and the catalogue
        is walked in chunks.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): List of arguments.
            item_indices (Optional[Tensor]): The batch of item indices. If None,
                full prediction will be produced.
            **kwargs (Any): The dictionary of keyword arguments.

        Returns:
            Tensor: The score matrix {user x item}.
        """
        if item_indices is None:
            items = torch.arange(self.n_items, device=user_indices.device)
            items = items.unsqueeze(0).expand(user_indices.size(0), -1)
        else:
            items = item_indices

        rings = self._rings_of(user_indices)

        # The candidates are read in blocks: the attention is {user x item x
        # memory}, which is small per item but not over a whole catalogue.
        block = 512
        scores: List[Tensor] = []
        for start in range(0, items.size(1), block):
            chunk = items[:, start : start + block]
            chunk_e = self.entity_embedding(self.entity_of(chunk))
            scores.append((chunk_e * self._address(rings, chunk_e)).sum(dim=-1))

        return torch.cat(scores, dim=1)
