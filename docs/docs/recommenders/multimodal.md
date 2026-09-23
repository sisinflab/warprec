# Multimodal Recommenders

The **Multimodal Recommenders** module of WarpRec contains models that score from **precomputed item features** alongside the interactions. Where a content-based model reads a sparse, hand-made attribute vector, these models read a dense representation produced by an encoder: a product photograph put through a convolutional network, a description put through a sentence encoder.

WarpRec reads those vectors and does not extract them. No model here fine-tunes an encoder, decodes an image or tokenises a description. What is read from the configuration stays as the file wrote it; what each model learns on top of it — a projection into the scoring space, and in some cases a refined copy of the vectors themselves — is set out below, model by model.

!!! info "API Reference"

    For class signatures, parameters, and source code, see the [Multimodal API Reference](../api-reference/recommenders/multimodal.md).

In the following sections, you will find the list of available multimodal models within WarpRec, together with their respective parameters. The features themselves are read as described in [Reading Multimodal Features](../data-management/reader.md#reading-multimodal-features).

## Data Requirements

Every model on this page requires `reader.multimodal` to be configured with at least one modality. Each modality names two files: the matrix of feature vectors and the row order that says which item each row describes. Configuring one of these models without features terminates the experiment during configuration validation.

An item that no row describes is kept in the catalogue with a vector of zeros and is scored from the collaborative half of the model alone.

## Choosing Modalities

Each model takes an optional `modalities` parameter naming which of the configured modalities it reads:

```yaml
models:
  VBPR:
    modalities: [visual]     # omit to read every configured modality
```

Left unset, a model reads **all** of them, so adding a modality to the reader does not leave it unused. Naming a modality the reader does not define fails during configuration validation, with the configured names listed.

## Summary of Available Multimodal Models

| Category | Model | Description |
|---|---|---|
| Embedding-Based | [VBPR](#vbpr) | Bayesian Personalized Ranking extended with a learned projection of the item features. |
| Graph-Based | [FREEDOM](#freedom) | A frozen item-item feature graph beside a denoised user-item graph. |

## Embedding-Based

Embedding-Based multimodal models project the features into the latent space and add the result to what the interactions already say about an item. There is no graph, which makes them the cheapest way to bring features into a model.

### VBPR

VBPR (Visual Bayesian Personalized Ranking): Represents an item as two things at once — a factor learned from who interacted with it, and a projection of what it looks like — and a user as two matching things, one reading each. A score is the dot product of the pair, which is the collaborative agreement plus the visual agreement, so the model falls back on the collaborative half for an item whose features are missing and leans on the feature half for an item almost nobody has touched. A per-item feature bias is carried beside the factors, so a look people generally prefer shifts every user's score alike. **This model requires multimodal features to function properly.**

Given several modalities it scores against their concatenation, which reduces exactly to the paper when only one is configured.

For further details, please refer to the [paper](https://arxiv.org/abs/1510.01784).

```yaml
models:
  VBPR:
    embedding_size: 64
    modalities: [visual]
    reg_weight: 0.00001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

- **embedding_size**: The width of both the collaborative and the projected feature factors. The user embedding is twice this, one half reading each.
- **The feature vectors are held fixed.** Only the projection out of them and the feature bias are learned, as in the paper.
- **modalities**: Which modalities to read. Defaults to every configured one.
- **reg_weight**: The L2 regularization weight.

## Graph-Based

Graph-Based multimodal models lay the features out as a graph over the items and propagate over it beside the user-item graph, so a signal can travel between items that look or read alike without any user having connected them.

### FREEDOM

FREEDOM: Builds two graphs and treats them differently. The **item-item graph**, built by nearest neighbours over the raw features of each modality, is *frozen*: computed once before training and never learned, on the argument that learning it costs a great deal and buys nothing. The **user-item graph** is *denoised*: a share of its edges is resampled at the start of every epoch, drawn against degree so the dense part of the graph is thinned rather than the sparse part, which stops the model depending on any single interaction. An item's representation is what its frozen neighbourhood says plus what propagation over the surviving interactions says, and each modality also enters the loss directly so its projection stays aligned with the collaborative space. **This model requires multimodal features to function properly.**

For further details, please refer to the [paper](https://dl.acm.org/doi/10.1145/3581783.3611943).

```yaml
models:
  FREEDOM:
    embedding_size: 64
    feature_size: 64
    knn_k: 10
    n_layers: 1
    n_ui_layers: 2
    dropout: 0.8
    modalities: [visual, textual]
    modality_weights: [0.1, 0.9]
    reg_weight: 0.00001
    batch_size: 2048
    epochs: 200
    learning_rate: 0.001
```

- **feature_size**: The width each modality is projected to before entering the loss.
- **The feature vectors are refined.** Each modality keeps a trainable copy, updated by the per-modality contrast, as the reference implementation does. The frozen copy the item-item graph was built from is never touched, so the graph stays the thing the paper froze.
- **knn_k**: How many neighbours each item keeps in the frozen graph.
- **n_layers**: How many hops to run over the frozen item-item graph.
- **n_ui_layers**: How many hops to run over the user-item graph.
- **dropout**: The share of interactions dropped at the start of each epoch. Set it to `0` to train on the whole graph.
- **modality_weights**: How much each modality's item-item graph counts, in the order `modalities` lists them. Defaults to equal weight.
- **reg_weight**: The weight of the per-modality contrast in the loss.

!!! note "Memory"

    The frozen graph is built from a similarity between every pair of items, which is quadratic in the catalogue. WarpRec takes that similarity in blocks and keeps only the `knn_k` nearest per item, so what is held is linear in the catalogue, but building it still costs one pass over the pairs.
