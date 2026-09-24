from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from narwhals.dataframe import DataFrame
from torch import Tensor

from warprec.utils.logger import logger


class KnowledgeGraph:
    """The facts a dataset carries about its items, as a graph of entities.

    Interactions are written about items and a knowledge graph is written about
    entities, so the two are joined by an alignment that says which entity each
    item stands for. An item without one keeps its interactions and simply has
    no facts attached; an entity without one is still part of the graph, because
    a fact two hops from an item is what the propagating models exist to reach.

    Everything here is held in index space, like the rest of the data layer: an
    entity is a row, a relation is a column of the relation embedding, and the
    models never see the identifiers the files were written with.

    Args:
        triples (DataFrame[Any]): The (head, relation, tail) facts.
        links (DataFrame[Any]): The (item, entity) alignment.
        item_mapping (dict): Mapping of item ID -> item index, from the dataset.
        head_label (str): The name of the head column.
        relation_label (str): The name of the relation column.
        tail_label (str): The name of the tail column.
        item_label (str): The name of the item column of the alignment.
        entity_label (str): The name of the entity column of the alignment.

    Attributes:
        n_entities (int): How many distinct entities the graph holds.
        n_relations (int): How many distinct relations the graph holds.
    """

    n_entities: int
    n_relations: int

    def __init__(
        self,
        triples: DataFrame[Any],
        links: DataFrame[Any],
        item_mapping: dict,
        head_label: str = "head",
        relation_label: str = "relation",
        tail_label: str = "tail",
        item_label: str = "item_id",
        entity_label: str = "entity_id",
    ):
        # pylint: disable = too-many-arguments, too-many-positional-arguments
        # The labels name the columns of two files the user wrote, and grouping
        # them would invent a structure neither file has.
        raw_heads = triples.select(head_label).to_numpy().flatten()
        raw_relations = triples.select(relation_label).to_numpy().flatten()
        raw_tails = triples.select(tail_label).to_numpy().flatten()

        entities = sorted({*raw_heads.tolist(), *raw_tails.tolist()})
        relations = sorted(set(raw_relations.tolist()))
        self._entity_index = {entity: i for i, entity in enumerate(entities)}
        self._relation_index = {relation: i for i, relation in enumerate(relations)}

        self.n_entities = len(self._entity_index)
        self.n_relations = len(self._relation_index)

        self._heads = self._to_index(raw_heads, self._entity_index)
        self._relations = self._to_index(raw_relations, self._relation_index)
        self._tails = self._to_index(raw_tails, self._entity_index)

        self._item_entity = self._align(links, item_mapping, item_label, entity_label)
        self._neighbour_index: Optional[Tuple[Tensor, Tensor, Tensor]] = None

        aligned = int((self._item_entity >= 0).sum())
        logger.stat_msg(
            f"Triples: {len(self._heads)}      Entities: {self.n_entities}      "
            f"Relations: {self.n_relations}      "
            f"Items with an entity: {aligned}/{len(self._item_entity)}",
            "Knowledge graph",
        )

    @staticmethod
    def _to_index(values: np.ndarray, index: Dict[Any, int]) -> Tensor:
        """Map raw identifiers onto their positions.

        Args:
            values (np.ndarray): The raw identifiers.
            index (Dict[Any, int]): The mapping to apply.

        Returns:
            Tensor: The positions, as a long tensor.
        """
        return torch.as_tensor(
            [index[value] for value in values.tolist()], dtype=torch.long
        )

    def _align(
        self,
        links: DataFrame[Any],
        item_mapping: dict,
        item_label: str,
        entity_label: str,
    ) -> Tensor:
        """Say which entity each item of the catalogue stands for.

        Args:
            links (DataFrame[Any]): The (item, entity) alignment.
            item_mapping (dict): Mapping of item ID -> item index.
            item_label (str): The name of the item column.
            entity_label (str): The name of the entity column.

        Returns:
            Tensor: One entity index per item index, -1 where the item has none.
        """
        aligned = torch.full((len(item_mapping),), -1, dtype=torch.long)

        raw_items = links.select(item_label).to_numpy().flatten().tolist()
        raw_entities = links.select(entity_label).to_numpy().flatten().tolist()

        unknown_entities = 0
        for item, entity in zip(raw_items, raw_entities):
            position = item_mapping.get(item)
            if position is None:
                # The alignment names items the catalogue does not hold, which
                # is normal: it is written for the whole graph, not for a split.
                continue

            index = self._entity_index.get(entity)
            if index is None:
                unknown_entities += 1
                continue

            aligned[position] = index

        if unknown_entities:
            logger.attention(
                f"The alignment names {unknown_entities} entities that appear in "
                "no triple. Those items carry no facts."
            )

        return aligned

    def __len__(self) -> int:
        return len(self._heads)

    def get_triples(self) -> Tuple[Tensor, Tensor, Tensor]:
        """The facts, in index space.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The heads, relations and tails.
        """
        return self._heads, self._relations, self._tails

    def get_dims(self) -> Tuple[int, int]:
        """The size of the entity and relation spaces.

        Returns:
            Tuple[int, int]: The number of entities and of relations.
        """
        return self.n_entities, self.n_relations

    def get_item_entities(self) -> Tensor:
        """The entity each item stands for.

        Returns:
            Tensor: One entity index per item index, -1 where the item has none.
        """
        return self._item_entity

    def neighbour_index(self) -> Tuple[Tensor, Tensor, Tensor]:
        """The graph as adjacency lists, read in both directions.

        A fact is a statement about both of its ends, so an entity's
        neighbourhood holds the tails it points at and the heads that point at
        it. The lists are returned in the compressed form a CSR matrix uses: the
        neighbours of entity ``e`` are the slice ``offsets[e]:offsets[e + 1]``.

        The result is built once and kept, because every model that walks the
        graph walks the same lists.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The offsets, the neighbour entities
                and the relation each neighbour was reached by.
        """
        if self._neighbour_index is None:
            heads = torch.cat([self._heads, self._tails])
            tails = torch.cat([self._tails, self._heads])
            relations = torch.cat([self._relations, self._relations])

            order = torch.argsort(heads, stable=True)
            counts = torch.bincount(heads, minlength=self.n_entities)

            offsets = torch.zeros(self.n_entities + 1, dtype=torch.long)
            offsets[1:] = torch.cumsum(counts, dim=0)

            self._neighbour_index = (offsets, tails[order], relations[order])

        return self._neighbour_index

    def sample_neighbours(
        self, how_many: int, generator: torch.Generator
    ) -> Tuple[Tensor, Tensor]:
        """Draw a fixed-size neighbourhood for every entity.

        A graph has neighbourhoods of wildly differing size and a model that
        gathers over them wants a rectangle, so each entity is given exactly
        ``how_many`` neighbours: sampled without replacement where it has enough
        and with replacement where it does not. An entity no fact mentions is
        made its own neighbour, which is what keeps a gather over it defined
        without a branch.

        Args:
            how_many (int): How many neighbours to draw per entity.
            generator (torch.Generator): The stream to draw from.

        Returns:
            Tuple[Tensor, Tensor]: The {entity x how_many} neighbour entities and
                the relation each was reached by.
        """
        offsets, neighbours, relations = self.neighbour_index()

        counts = offsets[1:] - offsets[:-1]
        # An isolated entity stands in for its own neighbourhood, so the draw
        # below is over one candidate rather than over none.
        safe_counts = counts.clamp(min=1)

        draw = torch.rand((self.n_entities, how_many), generator=generator)
        picked = (
            (draw * safe_counts.unsqueeze(1))
            .long()
            .clamp(max=(safe_counts - 1).unsqueeze(1))
        )
        positions = offsets[:-1].unsqueeze(1) + picked

        # Where an entity has no neighbour at all the position above points at
        # whatever follows it, so those rows are replaced wholesale.
        isolated = (counts == 0).unsqueeze(1)
        self_entity = torch.arange(self.n_entities).unsqueeze(1).expand(-1, how_many)

        sampled_entities = torch.where(isolated, self_entity, neighbours[positions])
        sampled_relations = torch.where(
            isolated, torch.zeros_like(self_entity), relations[positions]
        )

        return sampled_entities, sampled_relations

    def adjacency(
        self,
        values: Optional[Tensor] = None,
        size: Optional[int] = None,
        offset: int = 0,
    ) -> Tensor:
        """The graph as a sparse matrix, ready to be multiplied against.

        Propagating over a knowledge graph is a product against this matrix, and
        holding it sparsely is what keeps a graph of millions of facts from
        materialising one message per fact.

        Args:
            values (Optional[Tensor]): The weight of each triple. Defaults to
                ones, which is the unweighted graph.
            size (Optional[int]): The side of the square matrix. Defaults to the
                number of entities; a larger one leaves room for the rows a
                model may prepend, such as the users of a collaborative graph.
            offset (int): How far the entity rows are shifted, for the same
                reason.

        Returns:
            Tensor: The sparse adjacency.
        """
        if values is None:
            values = torch.ones(len(self._heads))

        side = size if size is not None else self.n_entities
        indices = torch.stack([self._heads + offset, self._tails + offset])

        return torch.sparse_coo_tensor(indices, values, (side, side)).coalesce()
