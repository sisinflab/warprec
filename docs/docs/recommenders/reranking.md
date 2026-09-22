# Re-Ranking

A model ranks by how likely a user is to interact with an item. That is not always
the same as the best *list*: ten near-identical films, or ten drama films for someone
who also watches comedy, can each be individually well predicted and collectively poor.

A **re-ranker** reorders the head of a ranking against a second objective. It runs
after the model and before anything reads the result, so the list a run reports and
the list it writes out are the same list.

```yaml
rerank:
    name: MMR
    pool: 100
    params:
        diversity: 0.3
```

- **name**: The re-ranker to apply, `MMR` or `Calibration`. Omitted, lists are ranked by score alone.
- **pool**: How many of the top candidates are reconsidered. Defaults to `100`.
- **params**: The re-ranker's own parameters.

!!! info "API Reference"

    For class signatures and source code, see the [Re-Ranking API Reference](../api-reference/recommenders/reranking.md).

!!! important "`pool` is a budget, not a preference"
    Both objectives are greedy and quadratic in the number of candidates, so they
    cannot run over a whole catalogue. Only the top `pool` items are reordered;
    everything below is left as the model ranked it. A cutoff deeper than the pool
    keeps the model's own ordering for the remainder, so nothing is ever lost.

Re-ranking needs item attributes, since both objectives order items by *what they
are*. A run that configures one without `reader.side` raises rather than silently
ranking by score.

## MMR

**Maximal Marginal Relevance.** At each step it takes the item that is both well
scored and unlike what the list already holds, which is what stops a ranking made of
ten variations on one film.

- **diversity**: How much of the objective is redundancy rather than relevance, in `[0, 1]`. `0` leaves the ranking untouched, `1` ignores the model. Defaults to `0.3`.

For further details, please refer to this [paper](https://dl.acm.org/doi/10.1145/290941.291025).

## Calibration

**Calibrated recommendations.** A model trained for accuracy drifts towards whatever a
user consumes most and drops the rest: someone who watches four fifths drama and one
fifth comedy is served ten dramas. Calibration asks that the proportions of the list
resemble the proportions of the history.

- **weight**: How much of the objective is calibration rather than relevance, in `[0, 1]`. Defaults to `0.5`.
- **smoothing**: How far the list distribution is pulled towards the user's before the divergence is taken, which keeps a genre the list has yet to cover from making it infinite. Defaults to `0.01`.

For further details, please refer to this [paper](https://dl.acm.org/doi/10.1145/3240323.3240372).

## Choosing the weight

Both re-rankers are governed by a single weight that decides how much of the objective
is the second goal rather than relevance. At `0` the ranking is the model's own; at `1`
the model's scores are ignored entirely. Values in between give up some accuracy for
some of the second objective, and which trade is worth making depends on the catalogue
and on what the list is for, so the weight is worth tuning like any other
hyper-parameter.

!!! note "Re-ranking applies to the results, not to model selection"
    The re-ranker runs on the evaluation a run reports and on the recommendations it
    writes. Validation *during* hyper-parameter search ranks by score alone, so trials
    are selected on the model's own ordering.
