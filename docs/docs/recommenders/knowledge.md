# Knowledge-Aware Recommenders

The **Knowledge-Aware Recommenders** module of WarpRec contains models that score from a **knowledge graph** alongside the interactions. Where a content-based model reads a flat list of item attributes, these models read a graph of facts: an item is an entity, an entity is related to other entities, and those entities are related in turn. That structure is what lets a signal travel from an item to a director, to another film by the same director, and back to an item the user has never seen.

!!! info "API Reference"

    For class signatures, parameters, and source code, see the [Knowledge-Aware API Reference](../api-reference/recommenders/knowledge.md).

In the following sections, you will find the list of available knowledge-aware models within WarpRec, together with their respective parameters. The graph itself is read as described in [Reading a Knowledge Graph](../data-management/reader.md#reading-a-knowledge-graph).

## Data Requirements

Every model on this page requires `reader.knowledge` to be configured. It names two files: the `(head, relation, tail)` triples, and the `(item, entity)` alignment that says which entity each catalogue item stands for. Configuring one of these models without a graph terminates the experiment during configuration validation.

An item the graph is silent about — one that is not aligned, or is aligned to an entity no triple mentions — is kept in the catalogue and scored from the collaborative half of the model alone. For the feature-based models such an item has no feature: in KaHFM its factor row starts at zero, so before training only its bias scores it, and BPR then trains the row like any other; KGFlex scores it zero.

## Training

A few behaviours are shared by the embedding, propagation and memory models on this page; the [feature-based](#feature-based) ones are described in their own section:

- **Two objectives are optimised together.** A recommendation loss over sampled `(user, positive, negative)` triples, and a knowledge loss over facts drawn from the graph. Both are stepped in the same pass, so a single `epochs` and `learning_rate` govern the whole model.
- **Facts are learned translationally.** A relation projects entities into a space of its own, where a true fact is contrasted against the same fact with a uniformly corrupted tail.
- **Two embedding widths are configured.** `embedding_size` is the width of the representations that produce a score; `kg_embedding_size` is the width of the space a relation projects into. They are independent.
- **Negatives are drawn by `training.negative_sampling`,** as for any other model trained on sampled negatives.

## Summary of Available Knowledge-Aware Models

| Category | Model | Description |
|---|---|---|
| Embedding-Based | [CKE](#cke) | Collaborative Knowledge base Embedding; adds a TransR entity factor to a matrix factorization item factor. |
| Propagation-Based | [KGAT](#kgat) | Knowledge Graph Attention Network; attentive propagation over a collaborative knowledge graph. |
| | [KGCN](#kgcn) | Knowledge Graph Convolutional Network; a sampled neighbourhood weighted per user. |
| | [KGIN](#kgin) | Knowledge Graph Intent Network; propagation split across learned user intents. |
| Memory-Based | [RippleNet](#ripplenet) | Preferences spread outward from a user's history across the graph. |
| Feature-Based | [KaHFM](#kahfm) | Knowledge-aware Hybrid Factorization Machines; one interpretable factor per graph feature, started at its TF-IDF. |
| | [KGFlex](#kgflex) | Sparse feature factorization over the features each user is expert about, chosen by information gain. |

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

### KGCN

KGCN (Knowledge Graph Convolutional Network): An item is described by the entities around it, but not every relation matters equally to every user. Someone who picks films by director should have the "directed by" edges count for more than the "genre" ones, and KGCN makes that explicit: the weight of an edge is the agreement between the user's embedding and the relation it was reached by, normalised over the neighbourhood. Each user therefore reads a graph of their own. The neighbourhood is sampled to a fixed size once before training, which is what lets the whole gather be a rectangle rather than a ragged walk. **This model requires a knowledge graph to function properly.**

For further details, please refer to the [paper](https://arxiv.org/abs/1904.12575).

```yaml
models:
  KGCN:
    embedding_size: 64
    neighbour_size: 8
    n_iter: 1
    aggregator: sum
    reg_weight: 0.00001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

- **neighbour_size**: How many neighbours each entity is given. An entity with more is sampled down, one with fewer is sampled with replacement, and one with none stands in for its own neighbourhood.
- **n_iter**: How many hops out from an item to read. The gather grows as `neighbour_size ** n_iter`, so two hops at size 8 already reads 64 entities per item.
- **aggregator**: How a node is combined with its neighbourhood: `sum`, `neighbour` or `concat`.

!!! note "Scoring cost"

    The edge weights depend on the user, so there is no single item matrix to multiply a batch of users against: every pair is walked. Full-catalogue ranking is correspondingly more expensive than for a model whose item representations are shared.

### KGIN

KGIN (Knowledge Graph Intent Network): Treats an interaction as the outcome of an **intent** rather than as a bare link. The model keeps a small set of intents, each a learned mixture over the relations of the graph, and a user's taste is read as a distribution over them: someone whose intent is "same director" reads the graph through the directing edges, someone whose intent is "same genre" through the genre ones. The intents are pushed apart from one another by an independence term, because two intents that say the same thing are one intent written twice. **This model requires a knowledge graph to function properly.**

For further details, please refer to the [paper](https://arxiv.org/abs/2102.07057).

```yaml
models:
  KGIN:
    embedding_size: 64
    n_factors: 4
    n_hops: 3
    node_dropout: 0.5
    mess_dropout: 0.1
    independence: distance
    ind_weight: 0.01
    reg_weight: 0.00001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

- **n_factors**: How many intents to keep.
- **n_hops**: How many hops to propagate over the facts and the interactions.
- **node_dropout**: The share of graph edges dropped on each pass.
- **mess_dropout**: The dropout applied to each hop's output.
- **independence**: How intents are pushed apart. `distance` is the distance correlation the paper uses; `cosine` is a cheaper alternative.
- **ind_weight**: The weight of that independence term.

### RippleNet

RippleNet: There is no learned user vector at all. What stands for a user is the set of facts reachable from the things they have already interacted with: the items themselves, then the entities one fact away, then two, like ripples spreading out from where a stone landed. Each ring is addressed against the candidate item — the facts that speak to it most are weighted highest — and the rings are summed into the vector the item is scored against. A plausibility term keeps the remembered facts holding together as facts, so the model is not free to carry preference along a relation that means nothing. **This model requires a knowledge graph to function properly.**

For further details, please refer to the [paper](https://arxiv.org/abs/1803.03467).

```yaml
models:
  RippleNet:
    embedding_size: 16
    n_hop: 2
    n_memory: 32
    kg_weight: 0.01
    reg_weight: 0.00001
    batch_size: 1024
    epochs: 200
    learning_rate: 0.001
```

- **n_hop**: How many rings to spread out from the history.
- **n_memory**: How many facts each ring holds. A user's reachable set grows very fast, so the rings are sampled down to this size.
- **kg_weight**: The weight of the fact-plausibility term.

!!! warning "Memory and width"

    A relation here is a **matrix**, not a vector, so the relation table holds `embedding_size ** 2` numbers per relation and the rings are addressed with a batched matrix product. A width that is unremarkable for the other models on this page is expensive for this one; the paper uses 16.

!!! note "Scoring cost"

    Like KGCN, the representation depends on the pair rather than on the item alone, so full-catalogue ranking walks every pair.

## Feature-Based

Feature-Based knowledge models neither embed the graph nor propagate over it. They read it as an attribute table: a **first-order feature** is a `(relation, tail)` pair leaving an item's entity, such as *directed by Kubrick*, and a **second-order feature** walks one fact further, `(relation, relation, tail)`, such as *directed by someone born in the UK*. Facts are read in the direction they were written, from the item outwards. A feature is kept only when at least `min_feature_items` items of the catalogue carry it.

Both models were developed at SisInfLab and first implemented in [Elliot](https://github.com/sisinflab/elliot). The WarpRec versions follow the papers where Elliot's code departs from them, as noted below.

### KaHFM

KaHFM (Knowledge-aware Hybrid Factorization Machines): A factorization model whose factors are not latent. There is one factor per first-order feature, so a user's factor is how much they care about *directed by Kubrick* and an item's is how much it is about it. An item starts at the TF-IDF of its features, normalised to unit length, and a user at the mean of the items in their training history; BPR then refines both, together with an item bias, and every factor stays tied to the feature it started from. Elliot's three variants (KaHFM, KaHFMBatch, KaHFMEmbeddings) differ only in how they are optimised; this one is trained in mini-batches with the configured optimizer. Elliot's user profile keeps only the last item carrying each feature; WarpRec uses the mean the paper defines. **This model requires a knowledge graph to function properly.**

For further details, please refer to the [ISWC 2019 paper](https://doi.org/10.1007/978-3-030-30793-6_3) and its [TKDE extension](https://doi.org/10.1109/TKDE.2020.3010215).

```yaml
models:
  KaHFM:
    min_feature_items: 10
    reg_weight: 0.0025
    bias_reg_weight: 0.0
    batch_size: 1024
    epochs: 100
    learning_rate: 0.001
```

- **min_feature_items**: How many items must carry a feature for it to become a factor. The width of the model is the number of features kept, so there is no `embedding_size`.
- **reg_weight**: The L2 regularization weight of the user and item factors.
- **bias_reg_weight**: The L2 regularization weight of the item biases.

!!! note "Reading the factors"

    The model keeps the feature each factor stands for in `feature_labels`, in the identifiers the graph was read with, so a user's largest factors name the features their recommendations lean on.

### KGFlex

KGFlex: A user is described by the few features their choices depend on. Before training, each user's items are set against as many items they did not take, drawn at random, and every feature is weighed by its **information gain** at telling the two apart. Only features with a positive gain are kept, up to `first_order_limit` first-order and `second_order_limit` second-order ones per user, and the gain becomes the fixed weight `k_uf` of that feature for that user. Each kept feature has a global embedding `g_f` and bias `b_f`, and each user a personal embedding `p_uf` for every feature they kept; an item is scored as `Σ k_uf (p_uf · g_f + b_f)` over the features it shares with the user, trained with BPR. Elliot's prediction paired a user's embeddings with the wrong features, it skipped samples whose negative shared no feature with the user, and it rejected a first-order limit without a second-order one; none of that is reproduced. **This model requires a knowledge graph to function properly.**

For further details, please refer to the [paper](https://arxiv.org/abs/2107.14290).

```yaml
models:
  KGFlex:
    embedding_size: 10
    first_order_limit: 100
    second_order_limit: 100
    min_feature_items: 10
    batch_size: 1024
    epochs: 50
    learning_rate: 0.005
```

- **first_order_limit**: How many first-order features each user keeps, by information gain. `-1` keeps every informative one, `0` none.
- **second_order_limit**: The same, for second-order features.
- **min_feature_items**: How many items must carry a feature for it to be considered.

!!! warning "Memory"

    A user holds one embedding per feature they kept, so the model has `users × features kept per user × embedding_size` personal parameters. With both limits at `-1` on a large graph that is far more than a matrix factorization of the same width; the limits are the knob.

!!! note "Sparse graphs"

    An item is scored only through the features it shares with the user, and there is no collaborative term to fall back on: an item that shares none scores zero, and a user for whom no feature is informative scores every item zero. How many such users there are is logged when the model is built. On a graph with only a few facts per item, most of the catalogue is out of a user's reach and the model ranks far below a collaborative one; it is meant for graphs that describe items richly. The personal embeddings are not regularized, as in the paper, so [`early_stopping`](../configuration/models.md) is worth configuring.
