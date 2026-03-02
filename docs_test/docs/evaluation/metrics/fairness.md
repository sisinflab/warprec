# Fairness

**Fairness metrics** aim to ensure that recommender systems provide **equitable recommendations** across different user groups, particularly those defined by sensitive attributes (e.g., gender, age, socioeconomic status). These metrics help detect and mitigate **disparate impact or treatment** in recommendation outcomes.

!!! info "API Reference"

    For class signatures and source code, see the [Fairness Metrics API Reference](../../api-reference/metrics/fairness.md).

## BiasDisparityBD

**Bias Disparity (BiasDisparityBD@K).** Measures the **difference in recommendation bias** between user groups, indicating how much one group is favored over another.

$$
\text{BD}(u, c) = \frac{\text{BR}(u, c) - \text{BS}(u, c)}{\text{BS}(u, c)}
$$

where $\text{BR}$ is the bias in recommendations and $\text{BS}$ is the bias in the training source. Positive values indicate bias amplification; negative values indicate bias reduction.

!!! note

    This metric requires the user to provide clustering information (i.e., user group definitions).

For further details, please refer to this [link](https://arxiv.org/pdf/1811.01461).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [BiasDisparityBD]
```

## BiasDisparityBR

**Bias Disparity - Bias Recommendations (BiasDisparityBR@K).** Quantifies the **disparity in the frequency of biased** (e.g., popular) items recommended to different user groups within their top-K recommendations.

$$
\text{BR}(u, c) = \frac{P_{\text{rec}}(u, c) / P_{\text{rec}}(u)}{P_{\text{global}}(c)}
$$

where $P_{\text{rec}}(u, c)$ is the proportion of items from cluster $c$ recommended to user cluster $u$, and $P_{\text{global}}(c)$ is the global proportion of items in cluster $c$.

!!! note

    This metric requires the user to provide clustering information.

For further details, please refer to this [link](https://arxiv.org/pdf/1811.01461).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [BiasDisparityBR]
```

## BiasDisparityBS

**Bias Disparity - Bias Scores (BiasDisparityBS).** Measures the **disparity in the average bias scores** of recommended items across user groups, assessing score-level bias.

$$
\text{BS}(u, c) = \frac{P_{\text{train}}(u, c)}{P_{\text{global}}(c)}
$$

where $P_{\text{train}}(u, c)$ is the proportion of interactions from user cluster $u$ with items in cluster $c$ in the training set.

!!! note

    This metric requires the user to provide clustering information.

For further details, please refer to this [link](https://arxiv.org/pdf/1811.01461).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [BiasDisparityBS]
```

## ItemMADRanking

**Item MAD Ranking (ItemMADRanking@K).** Computes the **Mean Absolute Deviation of item ranks** across user groups, measuring fairness in item exposure in rankings.

$$
\text{ItemMADRanking} = \frac{1}{\binom{m}{2}} \sum_{i=1}^{m} \sum_{j=i+1}^{m} \left| \bar{g}_i - \bar{g}_j \right|
$$

where $\bar{g}_c$ is the average discounted gain (DCG) for items in cluster $c$ and $m$ is the number of item clusters.

!!! note

    This metric requires the user to provide clustering information.

For further details, please refer to this [link](https://link.springer.com/article/10.1007/s11257-020-09285-1).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [ItemMADRanking]
```

## ItemMADRating

**Item MAD Rating (ItemMADRating@K).** Computes the **Mean Absolute Deviation of predicted item ratings** across user groups, assessing fairness in predicted preferences.

$$
\text{ItemMADRating} = \frac{1}{\binom{m}{2}} \sum_{i=1}^{m} \sum_{j=i+1}^{m} \left| \bar{r}_i - \bar{r}_j \right|
$$

where $\bar{r}_c$ is the average rating/score for items in cluster $c$.

!!! note

    This metric requires the user to provide clustering information.

For further details on the concept, please refer to this [link](https://dl.acm.org/doi/abs/10.1145/3269206.3271795).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [ItemMADRating]
```

## REO

**Ranking-based Equal Opportunity (REO@K).** Assesses whether relevant items are **ranked similarly across user groups**, ensuring fair visibility of relevant content.

$$
\text{REO} = \frac{\sigma\bigl(P(R@K \mid g_1, y{=}1), \ldots, P(R@K \mid g_A, y{=}1)\bigr)}{\mu\bigl(P(R@K \mid g_1, y{=}1), \ldots, P(R@K \mid g_A, y{=}1)\bigr)}
$$

where $P(R@K \mid g_a, y{=}1)$ is the probability that a relevant item from group $g_a$ appears in the top-K recommendations.

!!! note

    This metric requires the user to provide clustering information.

For further details, please refer to this [paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401177).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [REO]
```

## RSP

**Ranking-based Statistical Parity (RSP@K).** Measures whether the **ranking positions of items** (regardless of relevance) are **equally distributed across user groups**, ensuring fairness in exposure.

$$
\text{RSP} = \frac{\sigma\bigl(P(R@K \mid g_1), \ldots, P(R@K \mid g_A)\bigr)}{\mu\bigl(P(R@K \mid g_1), \ldots, P(R@K \mid g_A)\bigr)}
$$

where $P(R@K \mid g_a)$ is the probability that an item from group $g_a$ is recommended in top-K.

!!! note

    This metric requires the user to provide clustering information.

For further details, please refer to this [paper](https://dl.acm.org/doi/abs/10.1145/3397271.3401177).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [RSP]
```

## UserMADRanking

**User MAD Ranking (UserMADRanking@K).** Measures the **Mean Absolute Deviation of item ranking positions** for each user group, focusing on rank consistency across users.

$$
\text{UserMADRanking} = \frac{1}{\binom{m}{2}} \sum_{i=1}^{m} \sum_{j=i+1}^{m} \left| \text{nDCG}_i - \text{nDCG}_j \right|
$$

where $\text{nDCG}_c$ is the average nDCG score for users in cluster $c$.

!!! note

    This metric requires the user to provide clustering information.

For further details, please refer to this [link](https://link.springer.com/article/10.1007/s11257-020-09285-1).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [UserMADRanking]
```

## UserMADRating

**User MAD Rating (UserMADRating@K).** Measures the **Mean Absolute Deviation of predicted item ratings** for each user group, capturing disparities in predicted relevance.

$$
\text{UserMADRating} = \frac{1}{\binom{m}{2}} \sum_{i=1}^{m} \sum_{j=i+1}^{m} \left| \bar{s}_i - \bar{s}_j \right|
$$

where $\bar{s}_c$ is the average of top-K recommendation scores for users in cluster $c$.

!!! note

    This metric requires the user to provide clustering information.

For further details on the concept, please refer to this [link](https://dl.acm.org/doi/abs/10.1145/3269206.3271795).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [UserMADRating]
```
