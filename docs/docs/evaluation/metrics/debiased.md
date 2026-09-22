# Debiased

**Debiased metrics** correct a ranking measure for the fact that an offline test set is **missing not at random**: an item is in it partly because the user liked it and partly because the logging system *showed* it. Each estimator divides a relevant item's contribution by the probability that it was observed, so that a hit on a rarely-shown item counts for more than a hit on one everybody saw.

They are reported alongside the naive metrics rather than replacing them, so a results table always shows what the correction changed. For the propensity model itself and the caveats that come with it, see [Debiased Evaluation](../debiased.md).

!!! info "API Reference"

    For class signatures and source code, see the [Debiased Metrics API Reference](../../api-reference/metrics/debiased.md).

!!! warning

    These metrics require `evaluation.propensity.estimator` to be set to `popularity`. Asking for one while the estimator is `uniform` raises an error rather than reporting an uncorrected number under a corrected name.

## IPSRecall

**Inverse Propensity Scored Recall (IPSRecall@K).** Recall@K with every retrieved relevant item weighted by the inverse of its propensity $p_i$. Unbiased, but with unbounded variance, so its value may exceed one.

$$
\text{IPSRecall@}K = \frac{1}{|S_u|} \sum_{i \in \text{top-}K} \frac{rel_i}{p_i}
$$

For further details, please refer to this [paper](https://dl.acm.org/doi/10.1145/3240323.3240355).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [IPSRecall]
    propensity:
        estimator: popularity
```

## SNIPSRecall

**Self-Normalized Inverse Propensity Scored Recall (SNIPSRecall@K).** The same correction, divided by the sum of the weights it actually used instead of by the number of relevant items. Slightly biased and far more stable, and bounded in $[0, 1]$.

$$
\text{SNIPSRecall@}K = \left. \sum_{i \in \text{top-}K} \frac{rel_i}{p_i} \middle/ \sum_{i \in S_u} \frac{1}{p_i} \right.
$$

For further details, please refer to this [paper](https://papers.nips.cc/paper_files/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [SNIPSRecall]
    propensity:
        estimator: popularity
```

## IPSDCG

**Inverse Propensity Scored Discounted Cumulative Gain (IPSDCG@K).** Discounted gain with each gain weighted by the inverse of its propensity.

$$
\text{IPSDCG@}K = \sum_{j=1}^{K} \frac{rel_j}{p_j \log_2 (j + 1)}
$$

The gain is deliberately left unnormalised. nDCG divides by an ideal DCG that does not depend on the propensities, so correcting only the numerator would give a number that is neither bounded by one nor the estimator the literature defines — which is why there is no `IPSnDCG`.

For further details, please refer to this [paper](https://dl.acm.org/doi/10.1145/3240323.3240355).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [IPSDCG]
    propensity:
        estimator: popularity
```

## SNIPSDCG

**Self-Normalized Inverse Propensity Scored Discounted Cumulative Gain (SNIPSDCG@K).** IPSDCG@K divided by the summed weights, which bounds it the same way SNIPSRecall is bounded.

$$
\text{SNIPSDCG@}K = \left. \text{IPSDCG@}K \middle/ \sum_{i \in S_u} \frac{1}{p_i} \right.
$$

For further details, please refer to this [paper](https://papers.nips.cc/paper_files/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [SNIPSDCG]
    propensity:
        estimator: popularity
```
