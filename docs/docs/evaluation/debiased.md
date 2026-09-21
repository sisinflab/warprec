# Debiased Evaluation

An offline test set is **missing not at random**. An item ends up in it partly because
the user liked it and partly because the system that collected the data *showed* it.
A metric that ignores the second reason rewards a model for reproducing whatever the
logging policy already preferred, which in practice means popularity.

Debiased estimators correct for this by weighting each relevant item by the inverse of
the probability that it was observed.

## The propensity

The probability that item $i$ was observed is estimated from how often it was
interacted with in the training data:

$$p_i = \left(\frac{n_i}{\max_j n_j}\right)^{\text{power}}$$

floored at `clip`. It is configured once, under `evaluation`:

```yaml
evaluation:
    propensity:
        estimator: popularity
        power: 0.5
        clip: 0.1
```

The floor matters. The estimators divide by $p_i$, so without it a single rarely-shown
item would dominate the result, and an item with no training interactions at all would
divide by zero.

Setting `power: 0` makes every propensity 1, which turns each debiased estimator back
into its naive counterpart exactly. That is a useful thing to check on your own data.

## The estimators

WarpRec ships four, in two pairs. The `IPS` form is the plain inverse-propensity
estimator: unbiased, but with unbounded variance, which is why its value can exceed
one. The `SNIPS` form divides by the sum of the weights it actually used instead of by
the number of relevant items, which makes it very slightly biased and far more stable.
**`SNIPS` is usually the one to report.**

| Metric | Corrects | Bounded |
|--------|----------|---------|
| [`IPSRecall@k`](metrics/debiased.md#ipsrecall) | Recall | no |
| [`SNIPSRecall@k`](metrics/debiased.md#snipsrecall) | Recall | yes, $[0, 1]$ |
| [`IPSDCG@k`](metrics/debiased.md#ipsdcg) | Discounted gain | no |
| [`SNIPSDCG@k`](metrics/debiased.md#snipsdcg) | Discounted gain | yes |

Their definitions and references are on the [Debiased metrics](metrics/debiased.md)
page. Report them alongside the naive metrics rather than in place of them.

```yaml
evaluation:
    metrics: [nDCG, Recall, IPSRecall, SNIPSRecall, SNIPSDCG]
    propensity:
        estimator: popularity
```

!!! warning
    Asking for a debiased metric while `estimator` is `uniform` raises an error. A
    number labelled `IPSRecall` that carries no correction is worse than no number.

## Caveats

- **The propensity model is an assumption, not a measurement.** Popularity is a proxy
  for exposure, and a good one only when the logging policy was mostly popularity
  driven. If you know the true exposure probabilities, they belong here instead.
- **Debiased numbers are not comparable with naive ones.** Report them as their own
  column, never as "our nDCG".
- **The correction cannot recover items that were never shown to anyone.** It reweights
  what the log contains; it does not invent what it does not.

## References

- Yang et al., *Unbiased Offline Recommender Evaluation for Missing-Not-At-Random
  Implicit Feedback*, RecSys 2018.
- Saito et al., *Unbiased Recommender Learning from Missing-Not-At-Random Implicit
  Feedback*, WSDM 2020.
- Swaminathan and Joachims, *The Self-Normalized Estimator for Counterfactual
  Learning*, NeurIPS 2015.
