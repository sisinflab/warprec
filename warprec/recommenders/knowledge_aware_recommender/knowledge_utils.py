from typing import Any, Optional, Tuple

import torch
from torch import Tensor, nn

from warprec.data.entities import KnowledgeGraph
from warprec.utils.logger import logger


class KnowledgeRecommenderUtils(nn.Module):
    """Common ground for the models that score from a knowledge graph.

    The graph is written about entities and the catalogue is written about items,
    so every model here needs the same two things: the facts to learn from, and
    the entity each item stands for. An item the graph says nothing about points
    at a padding row that stays zero, which lets it fall back on whatever the
    collaborative half of the model knows without a branch anywhere.

    Args:
        params (dict): Model parameters.
        info (dict): The dictionary containing dataset information.
        *args (Any): Variable length argument list.
        knowledge (Optional[KnowledgeGraph]): The facts about the items.
        **kwargs (Any): Arbitrary keyword arguments.

    Attributes:
        n_entities (int): The number of entities, excluding the padding row.
        n_relations (int): The number of relations.
        n_items (int): The number of items, from the recommender base class.
        item_entity (Tensor): The entity each item stands for, padding where none.
        triple_heads (Tensor): The head of each fact, in index space.
        triple_relations (Tensor): The relation of each fact, in index space.
        triple_tails (Tensor): The tail of each fact, in index space.

    Raises:
        ValueError: If no knowledge graph was provided.
    """

    n_entities: int
    n_relations: int

    # Set by the recommender base class; declared here so that the coverage
    # check below reads it as the integer it is.
    n_items: int

    # Registered buffers, annotated so that they read as the tensors they are
    # rather than as the union a buffer is typed with.
    item_entity: Tensor
    triple_heads: Tensor
    triple_relations: Tensor
    triple_tails: Tensor

    def __init__(
        self,
        params: dict,
        info: dict,
        *args: Any,
        knowledge: Optional[KnowledgeGraph] = None,
        **kwargs: Any,
    ):
        super().__init__(params, info, *args, **kwargs)  # type: ignore[call-arg]

        if knowledge is None:
            raise ValueError(
                f"{type(self).__name__} scores from a knowledge graph, but the "
                "dataset carries none. Configure 'reader.knowledge'."
            )

        self.n_entities = info["n_entities"]
        self.n_relations = info["n_relations"]
        self._graph = knowledge

        # The padding row is the entity of an item the graph is silent about. It
        # is held at zero, so such an item contributes nothing from this side.
        aligned = knowledge.get_item_entities().clone()
        aligned[aligned < 0] = self.n_entities
        self.register_buffer("item_entity", aligned)

        covered = int((knowledge.get_item_entities() >= 0).sum())
        if covered < 0.5 * self.n_items:
            logger.attention(
                f"The knowledge graph covers {covered} of {self.n_items} items. "
                "Models that read an item as its entity score the uncovered ones "
                "as one and the same, so a graph this partial will hurt them. "
                "Consider restricting the catalogue to the items it covers."
            )

        heads, relations, tails = knowledge.get_triples()
        self.register_buffer("triple_heads", heads)
        self.register_buffer("triple_relations", relations)
        self.register_buffer("triple_tails", tails)

    def entity_of(self, item: Tensor) -> Tensor:
        """The entity each of the given items stands for.

        Args:
            item (Tensor): The item indices.

        Returns:
            Tensor: The entity indices, padding where an item has none.
        """
        return self.item_entity[item.clamp(max=self.item_entity.numel() - 1)]

    def sample_triples(
        self, how_many: int, generator: torch.Generator
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Draw facts to learn from, each against a corrupted one.

        The corrupted tail is drawn uniformly, which is what the translational
        objectives contrast a true fact against. Drawing from a generator the
        caller owns keeps a run reproducible.

        Args:
            how_many (int): How many facts to draw.
            generator (torch.Generator): The stream to draw from.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The heads, relations, true
                tails and corrupted tails.
        """
        total = self.triple_heads.numel()
        chosen = torch.randint(0, total, (how_many,), generator=generator)

        corrupted = torch.randint(0, self.n_entities, (how_many,), generator=generator)

        return (
            self.triple_heads[chosen],
            self.triple_relations[chosen],
            self.triple_tails[chosen],
            corrupted.to(self.triple_tails.device),
        )
