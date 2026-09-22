# Cold-Start Protocols

The usual splitting strategies hold out **interactions**, leaving every user and item
with some history to learn from. That measures how well a model completes a history it
has already seen, which is not what a content-based or hybrid model exists for.

A **cold-start protocol** holds out **entities**. Every interaction of a sampled
fraction of the items — or of the users — goes to the evaluation set, so at evaluation
time the model is asked about something it has never seen.

## Splitting

```yaml
splitter:
    test_splitting:
        strategy: item_cold_start
        ratio: 0.1
```

- **item_cold_start** holds out every interaction of a fraction of the items.
- **user_cold_start** does the same for users.

!!! important
    `ratio` is a fraction of **entities**, not of interactions. Every other strategy
    takes a fraction of rows. How many rows end up in the evaluation set follows from
    how active the sampled entities happened to be.

WarpRec keeps the held-out entities in the catalogue automatically: they carry no
interaction, so they would otherwise never enter the mappings and could not be scored
at all. Nothing needs to be configured for that.

## Restricting the candidates

Ranking over the whole catalogue under a cold-start protocol mostly measures the warm
catalogue: the warm items are far more numerous and far better served by a
collaborative model, so they dominate the result.

```yaml
evaluation:
    candidates: cold
```

- **all** ranks the whole catalogue. The default.
- **cold** keeps only the items with no training interaction.
- **warm** keeps only the rest, which is useful for measuring what the protocol cost
  the models that were never meant to handle cold items.

`candidates` is item-side only. Under `user_cold_start` every item is warm, and the
restriction to cold users happens by itself because the evaluation set contains only
those users. Sampled evaluation ignores it, since it already supplies its own
candidates.

## Read the result against a baseline, not against zero

!!! warning "A restricted candidate pool has a random baseline, and it is not zero"

    A cold pool is much smaller than a catalogue, so retrieving `k` items from it
    gets a non-trivial share of them by chance alone. Run an unpersonalized baseline
    such as `Random` under the same protocol and compare against that, not against
    zero. WarpRec logs the pool size and the implied floor whenever a restriction is
    active.

    Ties are broken at random rather than by item id, which matters here more than it
    sounds: a purely collaborative model scores every cold item alike, because each is
    an all-zero column, so its ranking is decided entirely by the tie break. Deciding
    it by position would hand those models the lowest item ids, and in most catalogues
    those are the oldest and best-known entries. The shuffle is drawn once per batch,
    so it removes that bias without standing in for a random baseline of its own.

The models to expect a signal from under this protocol are the ones that score from
item attributes rather than from interactions. See
[Content-Based](../recommenders/content.md) and [Hybrid](../recommenders/hybrid.md).
