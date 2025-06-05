# 📈 [WarpRec](../../README.md) Evaluation

The `Evaluation` module within the WarpRec framework provides a comprehensive set of tools for rigorous model assessment.

## 📚 Table of Contents
- 📈 [Metrics](#📈-metrics)
- 🧮 [Evaluator](#🧮-evaluator)
    - 🧺 [Batch Evaluation](#🧺-batch-evaluation)
    - 🧱 [Metric Blocks](#🧱-metric-blocks)

## 📈 Metrics

The `Metrics` module offers a collection of state-of-the-art metrics for evaluating recommender systems. Users can easily configure and access these metrics via a configuration file. To optimize memory utilization and ensure high-speed computation, WarpRec performs metric calculations in batches, leveraging [PyTorch](https://pytorch.org/) tensors for efficient processing.

For a detailed exposition and a complete list of **built-in** metrics, refer to the [metrics documentation](metrics/README.md)

## 🧮 Evaluator

The `Evaluator` module is designed to align seamlessly with WarpRec's metric structure, incorporating advanced optimizations for demanding evaluation tasks. The subsequent sections delineate the operational principles of the evaluation process.

### 🧺 Batch Evaluation

The WarpRec `Evaluator` employs a **batch-oriented approach** for metric computation, significantly enhancing efficiency and throughput. In contrast, many existing frameworks utilize a **dictionary-based approach** for metric computation, as exemplified below:

```json
{
  "user_id_1": {
    "relevant_item_id_1": 1,
    "relevant_item_id_2": 1,
  },
  "user_id_2": {
    "relevant_item_id_30": 1,
    "relevant_item_id_15": 1,
  },
}
```

While this structure may suffice for small datasets, its computational cost escalates considerably with an increasing number of users or relevant items per user. WarpRec overcomes this limitation through two primary optimizations.

Firstly, WarpRec employs a **tensor-based data representation** instead of the dictionary approach. This facilitates highly efficient retrieval of top-k items.

Secondly, the inherent batching process optimizes metric computation. Since most recommender system metrics are evaluated **per-user**, instead of processing the entire rating matrix (which is often infeasible), WarpRec segments the data into batches. This approach drastically improves both processing speed and memory efficiency.

Furthermore, traditional frameworks often evaluate metrics sequentially, leading to redundant iterations over user data when dealing with large user bases. WarpRec mitigates this inefficiency by iterating through batches only once, computing **partial results** for each metric until the entire dataset has been processed.

### 🧱 Metric Blocks

Within the WarpRec evaluation module, **Metric Blocks** are defined as computationally intensive operations shared across multiple metrics. A prime example is the calculation of **relevant recommended items**. Performing this operation independently for each metric, would lead to significant computational overhead. WarpRec addresses this by computing such shared operations only **once per batch**, and then distributing the results to all relevant metrics. This approach significantly optimizes the overall evaluation process.

Some examples of **Metric Blocks** are:
- **Top-k Indices and Values**: Identifying the indices and corresponding values of the top k recommended items.
- **Binary/Discounted Relevance**: Determining item relevance, either as a binary (0/1) or discounted (e.g., using a logarithmic scale) value.
- **Valid Users**: Identifying users who possess at least one relevant item, which is crucial for accurate per-user metric calculations.
- Other foundational computations required by various metrics.
