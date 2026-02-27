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

## Sampled Evaluation

WarpRec also supports **sampled evaluation** for metrics. In this approach, instead of evaluating the model performance across the entire set of items, the framework samples a fixed number of negative items for each user.

Thus, for each user, the evaluation is performed over a reduced set of items composed of the true positives and the sampled negatives, i.e. $(positives + negatives)$.

However, this introduces a challenge: what happens if two or more users within the same batch have a different number of positive items?

To illustrate this issue, let us start with a toy example. Consider a dataset with 10 items (for simplicity of visualization) and a batch size of 5. The prediction tensor is shown below:

$$
Pred_{\text{full}} = \begin{bmatrix}
   0.95 & 0.12 & 0.44 & 0.77 & 0.05 & 0.81 & 0.50 & -\infty & 0.69 & 0.21 \\
   0.10 & 0.88 & -\infty & 0.29 & 0.73 & -\infty & 0.91 & 0.03 & 0.47 & 0.62 \\
   0.58 & 0.25 & 0.99 & 0.14 & 0.83 & 0.37 & 0.60 & 0.07 & -\infty & 0.40 \\
   0.71 & 0.01 & 0.35 & 0.90 & -\infty & 0.52 & 0.20 & 0.85 & 0.49 & 0.66 \\
   0.18 & 0.65 & 0.30 & 0.87 & 0.54 & 0.09 & 0.78 & -\infty & 0.23 & 0.93
\end{bmatrix}
$$

$$
Target_{\text{full}} = \begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

In the case of full evaluation, interactions already observed during training are masked in the prediction tensor using $-\infty$. This ensures that evaluation only considers the ranking of unseen items.

Suppose now we perform sampled evaluation with 2 negative samples per user. The resulting prediction tensor is significantly smaller:

$$
Pred_{\text{sampled}} = \begin{bmatrix}
    0.95 & -\infty & 0.12 & 0.44 \\
    0.88 & 0.91 & 0.29 & 0.47 \\
    0.99 & 0.83 & 0.14 & 0.25 \\
    0.90 & -\infty & 0.20 & 0.49 \\
    0.87 & -\infty & 0.23 & 0.18
\end{bmatrix}
$$

$$
Target_{\text{sampled}} = \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    1 & 1 & 0 & 0 \\
    1 & 1 & 0 & 0 \\
    1 & 0 & 0 & 0 \\
    1 & 0 & 0 & 0
\end{bmatrix}
$$

In this setting, $-\infty$ is used for **padding**. Since WarpRec requires rectangular tensors, the number of positive labels must be padded to match across users.

The final evaluation results will differ from those obtained with full evaluation, but the main advantages of sampled evaluation are **reduced computational cost and lower memory usage**.

!!! note

    WarpRec during the sampled evaluation applies a *random shuffling* of positives and negatives. This prevents any bias that could arise from the ordering of items in the sampled tensors. The shuffling is seeded for reproducibility, ensuring consistent results across multiple runs and *removing any ordering bias*. For simplicity, this is not shown in the equations above.

---

## Metric Blocks

Let us introduce the concept of a `Metric Block`. As explained in the [evaluation](../core/evaluation/index.md) section, WarpRec organizes evaluation by computing shared data structures -- referred to as Metric Blocks -- across different metrics.

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
| `VALID_USERS` | The number (or mask) of users that have at least one relevant item in the evaluation set. This block ensures that metrics are computed only on meaningful user subsets. Returns the number of valid users in the batch. |
| `TOP_K_INDICES` | The indices of the top-$k$ predictions returned by the model for each user. Dim: `[batch_size, top_k]`. |
| `TOP_K_VALUES` | The actual prediction scores of the top-$k$ items for each user, aligned with `TOP_K_INDICES`. Dim: `[batch_size, top_k]`. |
| `TOP_K_BINARY_RELEVANCE` | The binary relevance (`[0, 1]`) of the top-$k$ predicted items, used in precision, recall, and hit-rate computations. Dim: `[batch_size, top_k]`. |
| `TOP_K_DISCOUNTED_RELEVANCE` | The discounted relevance values of the top-$k$ predicted items, used in ranking-aware metrics such as nDCG. Dim: `[batch_size, top_k]`. |
