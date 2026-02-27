# Accuracy

**Accuracy metrics** quantify how well a recommender system predicts user preferences or identifies relevant items. They assess the **correctness** of recommendations by comparing predicted interactions or ratings against actual user behavior. High accuracy generally indicates that the system is effective at surfacing items users are likely to engage with.

## AUC (Area Under the ROC Curve)

:::warprec.evaluation.metrics.accuracy.auc

Measures the probability that a randomly chosen relevant item is ranked higher than a randomly chosen irrelevant one.

$$
\text{AUC} = \frac{1}{|\mathcal{P}|} \sum_{(u,i) \in \mathcal{P}} \frac{|\{j \in \mathcal{N}_u : \hat{r}_{ui} > \hat{r}_{uj}\}|}{|\mathcal{N}_u|}
$$

For further details, please refer to this [link](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

```yaml
evaluation:
    metrics: [AUC]
```

## F1-Score@K

The harmonic mean of Precision@K and Recall@K, providing a balanced measure of accuracy.

$$
\text{F1@}K = \frac{(1 + \beta^2) \cdot \text{Precision@}K \cdot \text{Recall@}K}{\beta^2 \cdot \text{Precision@}K + \text{Recall@}K}
$$

For further details, please refer to this [book](https://link.springer.com/chapter/10.1007/978-1-4899-7637-6_8>) and this [link](https://en.wikipedia.org/wiki/Precision_and_recall).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [F1]
```

**Extended-F1** is also available, allowing you to compute the harmonic mean of any two metrics of your choice, as follows:

```yaml
evaluation:
    top_k: [10, 20, 50]
    complex_metrics:
        - name: F1
            params:
                metric_name_1: nDCG
                metric_name_2: MAP
                beta: 0.5
```

## GAUC (Group Area Under the ROC Curve)

Computes AUC per user (or group), then averages the results; accounts for group-level ranking quality.

$$
\text{GAUC} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \text{AUC}_u
$$

For further details, please refer to this [paper](https://www.ijcai.org/Proceedings/2019/0319.pdf)

```yaml
evaluation:
    metrics: [GAUC]
```

## HitRate@K

Measures the percentage of users for whom at least one relevant item is found within the top K recommendations.

$$
\text{HitRate@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \mathbb{1}\left[|\mathcal{R}_u \cap \mathcal{L}_u^K| > 0\right]
$$

For further details, please refer to this [link](https://en.wikipedia.org/wiki/Hit_rate).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [HitRate]
```

## LAUC (Limited Area Under the ROC Curve)

AUC computed over a limited set of top-ranked items, focusing on ranking quality within the most relevant recommendations.

For further details, please refer to this [paper](https://wiki.epfl.ch/edicpublic/documents/Candidacy%20exam/Evaluation.pdf).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [LAUC]
```

## MAP@K (Mean Average Precision)

Measures the mean of average precision scores across all users, rewarding correct recommendations ranked higher.

$$
\text{MAP@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{\min(|\mathcal{R}_u|, K)} \sum_{i=1}^{K} \text{Precision@}i \cdot r_i
$$

For further details, please refer to this [link](https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [MAP]
```

## MAR@K (Mean Average Recall)

Measures the mean of average recall scores across all users, indicating how well the relevant items are retrieved on average.

$$
\text{MAR@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{\min(|\mathcal{R}_u|, K)} \sum_{i=1}^{K} \text{Recall@}i \cdot r_i
$$

For further details, please refer to this [link](https://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#So-Why-Did-I-Bother-Defining-Recall?).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [MAR]
```

## MRR@K (Mean Reciprocal Rank)

Measures the average of the reciprocal ranks of the first relevant item in the recommendations.

$$
\text{MRR@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{1}{\text{rank}_u}
$$

For further details, please refer to this [link](https://en.wikipedia.org/wiki/Mean_reciprocal_rank).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [MRR]
```

## NDCG@K (Normalized Discounted Cumulative Gain)

Evaluates the **ranking quality** of recommendations, giving higher scores to relevant items appearing at higher ranks.

$$
\text{DCG@}K = \sum_{i=1}^{K} \frac{2^{r_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}
$$

For further details, please refer to this [link](https://en.wikipedia.org/wiki/Discounted_cumulative_gain).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [nDCG]
```

**nDCGRendle2020** is also available, allowing you to compute nDCG on binary relevance.

For further details, please refer to this [link](https://dl.acm.org/doi/10.1145/3394486.3403226).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [nDCGRendle2020]
```

## Precision@K

Measures the proportion of recommended items at rank K that are actually relevant.

$$
\text{Precision@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{|\mathcal{R}_u \cap \mathcal{L}_u^K|}{K}
$$

For further details, please refer to this [link](https://en.wikipedia.org/wiki/Precision_and_recall).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [Precision]
```

## Recall@K

Measures the proportion of relevant items that are successfully recommended within the top K items.

$$
\text{Recall@}K = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \frac{|\mathcal{R}_u \cap \mathcal{L}_u^K|}{|\mathcal{R}_u|}
$$

For further details, please refer to this [link](https://en.wikipedia.org/wiki/Precision_and_recall).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [Recall]
```
