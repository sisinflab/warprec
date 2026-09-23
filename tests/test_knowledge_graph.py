"""Behavioural tests for the knowledge graph and the models that read it.

The graph is a join between two files the user wrote and a catalogue the
framework built, so the tests here are mostly about that join: which entity an
item ends up standing for, what happens to an item the graph never mentions, and
whether what the models are handed is still in the index space everything else
in the data layer speaks. The model tests ask the two things a knowledge-aware
model can get silently wrong: that it refuses a dataset without a graph rather
than scoring from nothing, and that its attention is a distribution that moves.
"""

from typing import Tuple

import narwhals as nw
import pandas as pd
import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.data.entities import KnowledgeGraph
from warprec.utils.registry import model_registry

from conftest import build_params

# The identifiers are deliberately not 0..n: a graph is written with whatever
# names its source used, and the mapping into index space is what is under test.
TRIPLES = pd.DataFrame(
    {
        "head": ["m1", "m1", "m2", "m3", "d_kubrick"],
        "relation": ["directed_by", "genre", "genre", "directed_by", "born_in"],
        "tail": ["d_kubrick", "drama", "drama", "d_wilder", "uk"],
    }
)

LINKS = pd.DataFrame(
    {
        # "m9" is not in the catalogue and "ghost" appears in no triple: both are
        # ordinary, and neither may reach the models.
        "item_id": ["m1", "m2", "m3", "m9", "m4"],
        "entity_id": ["m1", "m2", "m3", "m1", "ghost"],
    }
)

# "m4" is aligned to an entity no triple mentions and "m5" is not aligned at all.
ITEM_MAPPING = {"m1": 0, "m2": 1, "m3": 2, "m4": 3, "m5": 4}


@pytest.fixture(scope="module")
def graph() -> KnowledgeGraph:
    """The small hand-written graph the data-layer tests read.

    Returns:
        KnowledgeGraph: The graph built against ITEM_MAPPING.
    """
    return KnowledgeGraph(
        nw.from_native(TRIPLES, eager_only=True),
        nw.from_native(LINKS, eager_only=True),
        ITEM_MAPPING,
    )


def test_the_vocabularies_cover_every_identifier_once(graph: KnowledgeGraph):
    """Entities and relations are counted over both ends of every triple."""
    entities, relations = graph.get_dims()

    # m1, m2, m3, d_kubrick, d_wilder, drama, uk: heads and tails share a space.
    assert entities == 7
    assert relations == 3


def test_the_triples_are_translated_into_index_space(graph: KnowledgeGraph):
    """Nothing the models receive still carries a raw identifier."""
    heads, relations, tails = graph.get_triples()

    assert len(graph) == len(TRIPLES)
    for part in (heads, relations, tails):
        assert part.dtype == torch.long
        assert part.numel() == len(TRIPLES)

    entities, n_relations = graph.get_dims()
    assert int(heads.max()) < entities and int(heads.min()) >= 0
    assert int(tails.max()) < entities and int(tails.min()) >= 0
    assert int(relations.max()) < n_relations and int(relations.min()) >= 0


def test_the_same_identifier_is_the_same_entity_on_either_end(graph: KnowledgeGraph):
    """A head and a tail written alike must land on one row.

    The graph shares one space between the two ends, so "m1" as the subject of a
    fact and "m1" as the object of one are the same node. Indexing the two ends
    separately would silently double the entity space and cut every path in it.
    """
    heads, _, _ = graph.get_triples()
    items = graph.get_item_entities()

    # The first two triples are written about m1, which is also item 0.
    assert int(heads[0]) == int(heads[1]) == int(items[0])


def test_an_item_stands_for_the_entity_it_was_aligned_to(graph: KnowledgeGraph):
    """The alignment is applied in the catalogue's index space, not the file's."""
    items = graph.get_item_entities()
    heads, _, tails = graph.get_triples()

    assert items.numel() == len(ITEM_MAPPING)

    # m3 is item 2, and its only fact names d_wilder as the tail.
    assert int(heads[3]) == int(items[2])
    # m2 is item 1 and shares the genre tail with m1, so the two differ as heads
    # and agree on where that fact points.
    assert int(items[0]) != int(items[1])
    assert int(tails[1]) == int(tails[2])


def test_an_item_the_graph_is_silent_about_carries_no_entity(graph: KnowledgeGraph):
    """Two kinds of silence both end as -1, and neither drops the item."""
    items = graph.get_item_entities()

    # m4 was aligned to an entity that appears in no triple; m5 was never aligned.
    assert int(items[3]) == -1
    assert int(items[4]) == -1

    # The items are still part of the catalogue: a model may know them from the
    # interactions even when the graph does not.
    assert items.numel() == len(ITEM_MAPPING)


def test_an_alignment_beyond_the_catalogue_is_ignored(graph: KnowledgeGraph):
    """A file written for the whole graph may name items a split does not hold."""
    items = graph.get_item_entities()

    # "m9" is aligned to m1's entity but is not in the catalogue, so no item may
    # have picked it up: item 0 is the only one standing for that entity.
    heads, _, _ = graph.get_triples()
    assert int((items == int(heads[0])).sum()) == 1


def test_the_adjacency_is_square_and_holds_one_cell_per_fact(graph: KnowledgeGraph):
    """The sparse layout is what keeps a large graph from being materialised."""
    adjacency = graph.adjacency()
    entities, _ = graph.get_dims()

    assert adjacency.is_sparse
    assert adjacency.shape == (entities, entities)
    assert adjacency.indices().size(1) == len(TRIPLES)

    heads, _, tails = graph.get_triples()
    indices = adjacency.indices()
    written = set(zip(indices[0].tolist(), indices[1].tolist()))
    assert written == set(zip(heads.tolist(), tails.tolist()))


def test_the_adjacency_can_be_shifted_to_leave_room_for_other_nodes(
    graph: KnowledgeGraph,
):
    """A collaborative graph prepends its users, so the entities move down."""
    entities, _ = graph.get_dims()
    offset = 11

    adjacency = graph.adjacency(size=entities + offset, offset=offset)

    assert adjacency.shape == (entities + offset, entities + offset)
    # Every entity row has moved by exactly the offset, and nothing landed in
    # the rows the caller reserved for itself.
    assert int(adjacency.indices().min()) >= offset

    plain = graph.adjacency()
    assert torch.equal(adjacency.indices(), plain.indices() + offset)


def test_the_adjacency_carries_the_weights_it_is_given(graph: KnowledgeGraph):
    """The attention a model learns is written onto the edges through here."""
    weights = torch.arange(1, len(TRIPLES) + 1, dtype=torch.float)

    adjacency = graph.adjacency(values=weights)

    assert pytest.approx(float(adjacency.values().sum())) == float(weights.sum())


def test_a_knowledge_model_refuses_a_dataset_without_a_graph(dataset: Dataset):
    """Scoring from a graph that is not there would quietly become collaborative."""
    for name in ("CKE", "KGAT"):
        with pytest.raises(ValueError, match="knowledge graph"):
            model_registry.get(
                name,
                params=build_params(name),
                info=dataset.info(),
                interactions=dataset.train_set,
                sessions=dataset.train_session,
                transactions=dataset.train_transactions,
                knowledge=None,
            )


def test_an_unaligned_item_points_at_the_padding_row(dataset: Dataset):
    """The padding entity is what lets an unknown item skip a branch."""
    model = model_registry.get(
        "CKE",
        params=build_params("CKE"),
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
        knowledge=dataset.knowledge,
    )

    entities = model.entity_of(torch.arange(dataset.info()["n_items"]))

    # Nothing is left negative, and nothing points past the padding row.
    assert int(entities.min()) >= 0
    assert int(entities.max()) <= model.n_entities


def test_the_attention_is_a_distribution_over_each_neighbourhood(dataset: Dataset):
    """Softmax over a sparse row is what weights a hop; it must stay normalised."""
    model = model_registry.get(
        "KGAT",
        params=build_params("KGAT"),
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
        knowledge=dataset.knowledge,
    )

    model.refresh_attention()
    rows = torch.sparse.sum(model.attention, dim=1).to_dense()

    # Every node with an edge sums to one; a node without edges sums to zero.
    populated = rows[rows > 0]
    assert populated.numel() > 0
    assert torch.allclose(populated, torch.ones_like(populated), atol=1e-5)


def test_the_attention_follows_the_embeddings_it_is_computed_from(dataset: Dataset):
    """A refresh that changed nothing would leave the model a fixed-weight GNN."""
    model = model_registry.get(
        "KGAT",
        params=build_params("KGAT"),
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
        knowledge=dataset.knowledge,
    )

    model.refresh_attention()
    before = model.attention.values().clone()

    with torch.no_grad():
        model.node_embedding.weight.mul_(2.5)
    model.refresh_attention()

    assert not torch.allclose(before, model.attention.values())


def test_the_dataset_publishes_the_dimensions_the_models_size_themselves_from(
    dataset: Dataset, knowledge_frames: Tuple[pd.DataFrame, pd.DataFrame]
):
    """A model reads the entity and relation counts out of the dataset info."""
    triples, _ = knowledge_frames
    info = dataset.info()

    assert dataset.knowledge is not None
    assert (info["n_entities"], info["n_relations"]) == dataset.knowledge.get_dims()

    # The counts are the graph's own, not the catalogue's: the fixture reaches
    # past the items, and a model sized from the catalogue would index out of it.
    assert info["n_entities"] == len({*triples["head"], *triples["tail"]})
    assert info["n_relations"] == triples["relation"].nunique()
    assert info["n_entities"] > info["n_items"]
