# Training Configuration

The **Training Configuration** module holds the options that shape the *signal* a
model learns from. They are separate from the [Reader](reader.md) on purpose: the
dataset the reader produces is identical whichever value they take, and only the
examples drawn from it during training differ.

## Available Keywords

- **negative_sampling**: How negative examples are drawn during training: `uniform` or `popularity`. Defaults to `uniform`, which gives every item the same chance. `popularity` draws proportionally to a dampened interaction count, producing harder negatives among the items a model is most likely to over-recommend.
- **neg_alpha**: The exponent `popularity` applies to the interaction counts before drawing, `count^alpha`. Defaults to `0.75`. Zero weighs every item equally, which recovers `uniform`; one draws in exact proportion to the counts, and higher concentrates the draw on the head. Must not be negative, and is ignored by `uniform`.

!!! note
    Both keywords drive the pointwise and pairwise (contrastive) samplers, which the context-aware models train through as well. The sequential models' loaders ignore them and always draw negatives uniformly among the items the user has not interacted with.

- **sequence_pooling**: How the values of a multi-valued contextual field are combined into the single vector that field contributes: `mean`, `sum` or `max`. Defaults to `mean`, which matches the normalised multi-hot encoding the factorization-machine literature defines these models over, so a field's contribution does not grow with the number of values it happens to hold.

!!! note
    A multi-valued field is declared with `reader.dtypes.context_separators`. Without
    one, `sequence_pooling` has nothing to pool and is ignored.

## Example Configuration

```yaml
training:
    negative_sampling: popularity
    sequence_pooling: mean
```

## Moved From the Reader

Both keywords were first released under `reader`. A configuration that still writes
them there keeps working: the value is carried over and a warning points at the new
location. An entry written under `training` always wins over the deprecated one.

```yaml
# Still accepted, but deprecated
reader:
    negative_sampling: popularity

# Preferred
training:
    negative_sampling: popularity
```
