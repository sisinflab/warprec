# Diversity

**Diversity metrics** evaluate the **variety of items** within a user's recommendations or across the recommendations for a set of users. These metrics are crucial for preventing "**filter bubbles**" and ensuring that users are exposed to a broad range of items, potentially increasing serendipity and user satisfaction.

!!! info "API Reference"

    For class signatures and source code, see the [Diversity Metrics API Reference](../../api-reference/metrics/diversity.md).

## Gini

**Gini Index.** Measures the **inequality** in the distribution of recommended items; lower values indicate more equitable item exposure.

$$
\text{Gini} = \frac{\sum_{j=1}^{|\mathcal{I}|} (2j - |\mathcal{I}| - 1) \cdot c_j}{(|\mathcal{I}| - 1) \cdot \sum_{j} c_j}
$$

where $c_j$ is the recommendation count for item $j$ (sorted in ascending order).

For further details, please refer to this [book](https://link.springer.com/rwe/10.1007/978-1-4939-7131-2_110158).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [Gini]
```

## ShannonEntropy

**Shannon Entropy.** Quantifies the **diversity** of recommended items using information entropy; higher values reflect greater item variety.

$$
H = -\sum_{i \in \mathcal{I}} p_i \log p_i, \quad p_i = \frac{c_i}{\sum_{j} c_j}
$$

For further details, please refer to this [book](https://link.springer.com/referenceworkentry/10.1007/978-1-4939-7131-2_110158).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [ShannonEntropy]
```

## SRecall

**Subtopic Recall (SRecall@K).** Measures how many **distinct subtopics or categories** are covered in the recommendations compared to the relevant ones, which reflects diversity across semantic dimensions.

!!! note

    This metric requires the user to provide side information (e.g., item categories).

For further details, please refer to this [paper](https://dl.acm.org/doi/abs/10.1145/2795403.2795405).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [SRecall]
```
