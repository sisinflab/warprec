# Coverage

**Coverage metrics** assess the extent to which a recommender system is able to recommend items from the entire catalog. They measure the **diversity** of the items recommended and the **proportion of the item space** that the system can effectively explore. High coverage suggests that the system can offer a wide range of recommendations beyond just the most popular items.

!!! info "API Reference"

    For class signatures and source code, see the [Coverage Metrics API Reference](../../api-reference/metrics/coverage.md).

## ItemCoverage

**Item Coverage (ItemCoverage@K).** Measures the **number of unique items** recommended in the top-K positions across all users, indicating **catalog coverage**.

$$
\text{ItemCoverage@}K = \left|\bigcup_{u \in \mathcal{U}} \mathcal{L}_u^K\right|
$$

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [ItemCoverage]
```

## UserCoverage

**User Coverage (UserCoverage@K).** Calculates the **number of users** with at least one recommended item in their top-K recommendations, indicating reach and usefulness.

$$
\text{UserCoverage@}K = |\{u \in \mathcal{U} : |\mathcal{L}_u^K| > 0\}|
$$

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [UserCoverage]
```

## NumRetrieved

**Number of Retrieved Items (NumRetrieved@K).** Counts the **total number of distinct items** retrieved in the top-K recommendations across all users.

For further details, please refer to this [link](https://github.com/RankSys/RankSys/blob/master/RankSys-metrics/src/main/java/es/uam/eps/ir/ranksys/metrics/basic/NumRetrieved.java).

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [NumRetrieved]
```

## UserCoverageAtN

**User Coverage At N (UserCoverageAtN).** Measures the number of users for whom the recommender retrieves at least **N** items, reflecting system responsiveness or minimum output capability.

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [UserCoverageAtN]
```
