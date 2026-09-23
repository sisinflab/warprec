# Knowledge-Aware Recommenders

The **Knowledge-Aware Recommenders** module of WarpRec contains models that score from a **knowledge graph** alongside the interactions. Where a content-based model reads a flat list of item attributes, these models read a graph of facts: an item is an entity, an entity is related to other entities, and those entities are related in turn. That structure is what lets a signal travel from an item to a director, to another film by the same director, and back to an item the user has never seen.

!!! info "API Reference"

    For class signatures, parameters, and source code, see the [Knowledge-Aware API Reference](../api-reference/recommenders/knowledge.md).

In the following sections, you will find the list of available knowledge-aware models within WarpRec, together with their respective parameters. The graph itself is read as described in [Reading a Knowledge Graph](../data-management/reader.md#reading-a-knowledge-graph).

## Data Requirements

Every model on this page requires `reader.knowledge` to be configured. It names two files: the `(head, relation, tail)` triples, and the `(item, entity)` alignment that says which entity each catalogue item stands for. Configuring one of these models without a graph terminates the experiment during configuration validation.

An item the graph is silent about — one that is not aligned, or is aligned to an entity no triple mentions — is kept in the catalogue and scored from the collaborative half of the model alone.

## Training

A few behaviours are shared by every model on this page:

- **Two objectives are optimised together.** A recommendation loss over sampled `(user, positive, negative)` triples, and a knowledge loss over facts drawn from the graph. Both are stepped in the same pass, so a single `epochs` and `learning_rate` govern the whole model.
- **Facts are learned translationally.** A relation projects entities into a space of its own, where a true fact is contrasted against the same fact with a uniformly corrupted tail.
- **Two embedding widths are configured.** `embedding_size` is the width of the representations that produce a score; `kg_embedding_size` is the width of the space a relation projects into. They are independent.
- **Negatives are drawn by `training.negative_sampling`,** as for any other model trained on sampled negatives.

## Summary of Available Knowledge-Aware Models

| Category | Model | Description |
|---|---|---|
| Embedding-Based | [CKE](#cke) | Collaborative Knowledge base Embedding; adds a TransR entity factor to a matrix factorization item factor. |
| Propagation-Based | [KGAT](#kgat) | Knowledge Graph Attention Network; attentive propagation over a collaborative knowledge graph. |

## Embedding-Based

Embedding-Based knowledge models learn a representation for each entity and use it to enrich the item representation. Nothing is propagated over the graph: the structure enters through what the entity embeddings are trained to satisfy.

### CKE

CKE (Collaborative Knowledge base Embedding): Represents an item as the sum of a collaborative factor, learned from the interactions, and the embedding of the entity it stands for, learned from the facts. The knowledge component is TransR: a relation projects head and tail entities into a relation-specific space, where a true fact is the one whose projected head plus relation lands nearest its projected tail. The two components are trained jointly, so an item with few interactions still carries the structure its entity sits in. **This model requires a knowledge graph to function properly.**

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/2939672.2939673).

```yaml
models:
  CKE:
    embedding_size: 64
    kg_embedding_size: 64
    reg_weight: 0.00001
    kg_reg_weight: 0.00001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

- **kg_embedding_size**: The width of the relation-specific space entities are projected into by TransR.
- **reg_weight**: The L2 regularization weight of the recommendation component.
- **kg_reg_weight**: The L2 regularization weight of the knowledge component.

## Propagation-Based

Propagation-Based knowledge models lay the interactions and the facts out as a single graph and pass messages over it. A user, an item and an attribute are all nodes, so a representation is built from a neighbourhood that spans both kinds of edge.

### KGAT

KGAT (Knowledge Graph Attention Network): Builds a **collaborative knowledge graph**, in which users are nodes beside the entities and an interaction is a relation of its own, and propagates over it. Each hop is a bi-interaction aggregator, which combines the sum and the element-wise product of a node with its aggregated neighbourhood. How much each edge carries is not fixed: an attention weight is computed from how well the edge's fact translates under its relation, normalised over each node's neighbourhood, and recomputed at the start of every epoch from the current embeddings. The output of every hop is concatenated, so representations at different distances all reach the score. **This model requires a knowledge graph to function properly.**

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/3292500.3330989).

```yaml
models:
  KGAT:
    embedding_size: 64
    kg_embedding_size: 64
    layers: [64, 32]
    dropout: 0.1
    reg_weight: 0.00001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

- **layers**: One width per propagation hop. The list length is the number of hops, so `[64, 32]` propagates twice.
- **kg_embedding_size**: The width of the relation-specific space used both by the knowledge loss and by the attention.
- **dropout**: Applied to the output of each aggregator.
- **reg_weight**: The L2 regularization weight, applied to both components.

!!! note "Memory"

    The collaborative knowledge graph holds an edge per interaction and per fact, in both directions. It is kept sparse, so propagation is a sparse product rather than one message per edge, but the graph still grows with the dataset and the number of triples.
