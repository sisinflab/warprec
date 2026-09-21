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

## Read the result against the floor, not against zero

!!! warning "A restricted candidate pool has a high random baseline"

    A cold pool is small. Retrieving 10 items from a pool of 24 gets you about 42% of
    them **by scoring them all identically**, and that is exactly what a purely
    collaborative model does: every cold item is an all-zero column, so it assigns them
    all the same score and the ranking is decided by tie-breaking.

    In a run on 120 items with 24 held out, ItemKNN scored `HitRate@10 = 0.7750`. The
    expected value from tie-breaking alone was `0.7790`. It had learned nothing.

    WarpRec logs the pool size and the implied floor whenever a restriction is active.
    Compare against that number, and against a random baseline you run yourself, before
    concluding that a model handles cold items well.

A model that genuinely uses item attributes — the content-based and hybrid families —
is the one to expect a real signal from here. See
[Content-Based](../recommenders/content.md) and [Hybrid](../recommenders/hybrid.md).
