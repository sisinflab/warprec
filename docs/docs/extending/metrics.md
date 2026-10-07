# Custom Metric Implementation Guide

In this section, we'll walk you through how to implement your own evaluation metric inside WarpRec.

## Metric Interfaces

First, import the main metric interfaces:

```python
from warprec.evaluation.metrics.base_metric import UserAverageTopKMetric, TopKMetric, RatingMetric, BaseMetric
```

The **BaseMetric** interface is sufficient for metrics that **do not depend on the cutoff (K)**.

If your metric is an error metric (e.g., MSE, MAE) that relies on raw ratings rather than ranking, use the **RatingMetric** interface.

If your metric is ranking-based and requires a cutoff, use the **TopKMetric** interface.

Moreover, if your metric needs to be computed on a per-user basis before aggregating to a system-wide value (e.g., average over users), use the **UserAverageTopKMetric** interface.

We'll use the **UserAverageTopKMetric** interface to implement a simple metric: **Precision@K**.

For a single user, Precision@K is the number of relevant items actually recommended divided by the cutoff K. The system-wide Precision@K is the sum of all individual Precision@K values divided by the number of users.

## The Implementation

Before starting the implementation, we need to define the variables we need to *accumulate* across batch iterations.

For **Precision@K**, we need to accumulate just one value: the number of **hits** (relevant items recommended) per user. This value will be stored in a tensor of length **num_users** where each index corresponds to the number of hits for that given user.

Using the **UserAverageTopKMetric** this step is *extremely* simplified, we only need to implement the logic to retrieve the number of hits from the top-k relevant item tensor, which can be implemented like this:

```python
def compute_scores(
    self, preds: Tensor, target: Tensor, top_k_rel: Tensor, **kwargs: Any
) -> Tensor:
    return top_k_rel.sum(dim=1).float() / self.k
```

In this implementation, `top_k_rel` is a binary tensor of shape [batch_size, k] where each entry is 1 if the recommended item at that position is relevant for the user and 0 otherwise. By summing over the second dimension (k), we get the total number of hits for each user in the batch. Finally, we divide by `self.k` to get the Precision@K for each user.

For the **UserAverageTopKMetric** interface, we don't need to worry about accumulating the hits across batches or computing the final average. The framework will automatically handle the accumulation of scores and the final averaging over users at the end of the evaluation process.

!!! note

    The number of users is not always fixed. We often only consider users who have **at least one relevant item** in the ground truth for evaluation. This ensures the denominator reflects only *valid* evaluation cases.

---

## Metric Blocks

Let us introduce the concept of a `Metric Block`. As explained in the [evaluation](../evaluation/index.md) section, WarpRec organizes evaluation by computing shared data structures -- referred to as Metric Blocks -- across different metrics.

This design does not increase the computational cost of evaluating individual metrics; instead, it enables the reuse of precomputed information, reducing redundancy and ensuring efficient data sharing throughout the entire evaluation process.

### Using Metric Blocks

To leverage a **Metric Block**, the implementation of a custom metric must be slightly refactored. The first step is to declare the required blocks within the metric class:

```python
from typing import Set
from warprec.utils.enums import MetricBlock

_REQUIRED_COMPONENTS: Set[MetricBlock] = {
    MetricBlock.BINARY_RELEVANCE,
    MetricBlock.VALID_USERS,
    MetricBlock.TOP_K_BINARY_RELEVANCE,
}
```

These components correspond to the same tensors that were explicitly computed in the previous implementation. However, when using Metric Blocks, WarpRec manages their computation and storage, and the metric only needs to retrieve them.

### Available Metric Blocks

WarpRec provides a set of **Metric Blocks**, i.e., reusable intermediate computations that can be shared across multiple metrics. By precomputing these components once and making them available to all metrics, evaluation becomes more efficient and avoids redundant tensor operations.

The following table summarizes the available Metric Blocks:

| **Metric Block** | **Description** |
|---|---|
| `BINARY_RELEVANCE` | Relevance encoded as a binary tensor `[0, 1]`, where 1 indicates that the item is relevant and 0 otherwise. Dim: `[batch_size, num_items]`. |
| `DISCOUNTED_RELEVANCE` | Relevance values adjusted by a discounting factor (e.g., logarithmic), typically used in ranking metrics such as nDCG. Dim: `[batch_size, num_items]`. |
| `VALID_USERS` | A mask of the users in the batch that have at least one relevant item in the evaluation set: 1 for such a user, 0 otherwise. User-averaged metrics use it so that a user with nothing to retrieve does not count. Dim: `[batch_size]`. |
| `TOP_K_INDICES` | The indices of the top-$k$ predictions returned by the model for each user. Dim: `[batch_size, top_k]`. |
| `TOP_K_VALUES` | The actual prediction scores of the top-$k$ items for each user, aligned with `TOP_K_INDICES`. Dim: `[batch_size, top_k]`. |
| `TOP_K_BINARY_RELEVANCE` | The binary relevance (`[0, 1]`) of the top-$k$ predicted items, used in precision, recall, and hit-rate computations. Dim: `[batch_size, top_k]`. |
| `TOP_K_DISCOUNTED_RELEVANCE` | The discounted relevance values of the top-$k$ predicted items, used in ranking-aware metrics such as nDCG. Dim: `[batch_size, top_k]`. |

!!! warning

    Declare every block your metric reads. The evaluator computes only the blocks some metric in the run asked for, so a metric that reads a block without declaring it works next to one that does, and fails when it is evaluated alone.

## Registering and Configuring a Metric

A metric becomes available by name once its class is registered and its module is imported. The name is what the configuration uses, case-insensitively:

```python
from warprec.utils.registry import metric_registry


@metric_registry.register("RBP")
class RBP(UserAverageTopKMetric):
    ...
```

Point `general.custom_modules` at the file (or package) that holds it. WarpRec imports it before the configuration is validated, so the metric can be named in either list of the `evaluation` section:

```yaml
general:
    custom_modules: [user_code/my_metrics.py]
evaluation:
    top_k: [10, 20]
    metrics: [nDCG, RBP]
    complex_metrics:
        - name: RBP
          params:
              persistence: 0.5
```

Every metric is constructed with the cutoff `k` and the keyword arguments the evaluator shares with all metrics (the number of users and items, the training matrix, the side information and cluster lookups, the propensities and the dataset's stash), so its constructor must accept `**kwargs`. The `params` of a `complex_metrics` entry arrive as extra keyword arguments. When one metric is configured with several sets of parameters, override its `name` property so that each appears under its own name in the results.

!!! tip

    [Guide 15 · Custom models and metrics](../guides/custom-models-and-metrics.md) builds a complete metric (rank-biased precision with a `persistence` parameter), uses it in both lists, and checks its values by hand.
