# Novelty

**Novelty metrics** assess the extent to which a recommender system suggests items that are **new or unexpected** to the user, beyond what is already popular or frequently consumed. These metrics are important for fostering **exploration and serendipity**, as highly novel recommendations can lead to delightful discoveries.

!!! info "API Reference"

    For class signatures and source code, see the [Novelty Metrics API Reference](../../api-reference/metrics/novelty.md).

## EFD

**Expected Free Discovery (EFD@K).** Estimates the likelihood that users discover relevant but less popular (**unexpected**) items in their top-K recommendations, promoting **serendipity**.

$$
\text{EFD@}K = \frac{1}{|\mathcal{U}| \cdot C} \sum_{u \in \mathcal{U}} \sum_{i=1}^{K} \frac{r_i \cdot (-\log_2 p_i)}{\log_2(i + 1)}
$$

where $p_i = n_i / |\mathcal{U}|$ is the popularity of item $i$ and $C = \sum_{i=1}^{K} 1/\log_2(i+1)$.

For further details, please refer to this [link](https://dl.acm.org/doi/abs/10.1145/2043932.2043955).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [EFD]
```

**Extended-EFD** is also available, meaning you can compute the EFD score using a discounted relevance value, as follows:

```yaml
evaluation:
    top_k: [10, 20, 50]
    complex_metrics:
        - name: EFD
          params:
              relevance: discounted
```

## EPC

**Expected Popularity Complement (EPC@K).** Measures the average **complement of item popularity** in the top-K recommendations, encouraging exposure to less popular content.

$$
\text{EPC@}K = \frac{1}{|\mathcal{U}| \cdot C} \sum_{u \in \mathcal{U}} \sum_{i=1}^{K} \frac{r_i \cdot (1 - p_i)}{\log_2(i + 1)}
$$

For further details, please refer to this [link](https://dl.acm.org/doi/abs/10.1145/2043932.2043955).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [EPC]
```

**Extended-EPC** is also available, meaning you can compute the EPC score using a discounted relevance value, as follows:

```yaml
evaluation:
    top_k: [10, 20, 50]
    complex_metrics:
        - name: EPC
          params:
              relevance: discounted
```
