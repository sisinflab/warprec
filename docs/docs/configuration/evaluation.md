# Evaluation Configuration

The **Evaluation Configuration** module defines the metrics and evaluation strategy for each model trained in WarpRec.
It provides flexible control over ranking cutoffs, sampling strategies, statistical significance tests, and reporting options.

!!! important
    Only users with relevant items are considered during evaluation. If the splitting strategy does not provide validation or test data for a given user, that user will be excluded from this step.

## Available Keywords

- **top_k**: Cutoff values used to compute ranking metrics. Can be a single integer or a list.
- **metrics**: List of evaluation metrics to compute, e.g., `nDCG`, `Precision`, `Recall`, `HitRate`.
- **complex_metrics** List of metrics and their parameters. Check the metrics documentation for more information.
- **validation_metric**: Metric used for model validation during training. Defaults to `nDCG@5`.
- **batch_size**: Batch size used during evaluation. Defaults to `1024`.
- **strategy**: Evaluation strategy: `full` or `sampled`. `sampled` is recommended for large datasets. Defaults to `full`.
- **num_negatives**: Number of negative samples used in the `sampled` strategy.

- **candidates**: Which items a run may rank: `all`, `cold` or `warm`. Defaults to `all`. `cold` keeps only the items with no training interaction, which is what makes a cold-start protocol measure cold-start rather than the warm catalogue around it. See [Cold-Start Protocols](../evaluation/cold-start.md).
- **mask_seen**: Which already-seen items are excluded from the ranking, under the `full` strategy: `auto`, `context`, `pair` or `none`. Defaults to `auto`, which excludes items the user has seen **in the same context** when the dataset has contextual columns, and every item the user has seen otherwise. `context` and `pair` force either behaviour and `none` excludes nothing.

    This setting also governs the recommendations written to disk, so the list a run reports and the list it writes are filtered by the same rule. A contextual rule is the one exception: a recommendation is produced for a user rather than for a user in a situation, so there is no context to compare a history against and the written list falls back to excluding every item the user has seen. WarpRec says so when it happens.
- **seed**: Random seed used for reproducibility in sampling. Defaults to `42`.
- **propensity**: Nested section defining the propensity model the debiased estimators read. See [Debiased Evaluation](../evaluation/debiased.md).
- **stat_significance**: Nested section defining statistical significance tests.
- **full_evaluation_on_report**: Whether to perform full evaluation each epoch. Defaults to `False`.
- **max_metric_per_row**: Number of metrics logged per row. Defaults to `4`.
- **save_evaluation**: Whether to save evaluation results. Defaults to `True`.
- **save_per_user**: Whether to save per-user evaluation results. Defaults to `False`.

## Propensity

The `propensity` section defines how WarpRec estimates the probability that each item
was *observed*, which the debiased estimators divide by. It only affects the metrics
that ask for it; every other metric is untouched.

- **estimator**: How the probability is estimated: `popularity` reads it off the training interaction counts, `uniform` applies no correction. Defaults to `uniform`.
- **power**: The exponent applied to the normalised counts, `p_i = (n_i / max_j n_j) ** power`. Lower values flatten the correction and `0` removes it entirely. Defaults to `0.5`.
- **clip**: The smallest propensity any item may carry. Defaults to `0.1`. The estimators divide by the propensity, so this floor is what stops one rarely-shown item from dominating the result, and what keeps an item with no interactions from dividing by zero.

!!! warning
    A debiased metric asked for while `estimator` is `uniform` raises rather than
    silently reporting an uncorrected number, because a result labelled `IPSRecall`
    that carries no correction is worse than no result at all.

## Statistical Significance

The **stat_significance** nested section allows users to configure statistical tests for evaluating metric differences:

- **paired_t_test**: Enable the Paired t-test. Defaults to `False`.
- **wilcoxon_test**: Enable the Wilcoxon signed-rank test. Defaults to `False`.
- **kruskal_test**: Enable the Kruskal-Wallis H-test. Defaults to `False`.
- **whitney_u_test**: Enable the Mann-Whitney U test. Defaults to `False`.
- **corrections**: Nested section defining corrections for multiple hypothesis testing.

## Corrections

The **corrections** section specifies methods to control family-wise error rate or false discovery rate:

- **bonferroni**: Apply Bonferroni correction. Defaults to `False`.
- **holm_bonferroni**: Apply Holm-Bonferroni correction. Defaults to `False`.
- **fdr**: Apply False Discovery Rate (FDR) correction. Defaults to `False`.
- **alpha**: Significance level (alpha) for hypothesis testing. Defaults to `0.05`.

## Example Evaluation Configuration

The following example evaluates the best model trained in the current iteration, using sampled evaluation and statistical tests:

```yaml
evaluation:
    top_k: [10, 20, 50]
    metrics: [nDCG, Precision, Recall, HitRate]
    strategy: sampled
    num_negatives: 999
    stat_significance:
        wilcoxon_test: True
        paired_t_test: True
        corrections:
            bonferroni: True
```
