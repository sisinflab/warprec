.. _evaluation_configuration:

#########################
Evaluation Configuration
#########################

The **Evaluation Configuration** module defines the metrics and evaluation strategy for each model trained in WarpRec.
It provides flexible control over ranking cutoffs, sampling strategies, statistical significance tests, and reporting options.

.. important::
    Only users with relevant items are considered during evaluation. If the splitting strategy does not provide validation or test data for a given user, that user will be excluded from this step.

-----------------------------
Available Keywords
-----------------------------

- **top_k**: Cutoff values used to compute ranking metrics. Can be a single integer or a list.
- **metrics**: List of evaluation metrics to compute, e.g., ``nDCG``, ``Precision``, ``Recall``, ``HitRate``.
- **validation_metric**: Metric used for model validation during training. Defaults to ``nDCG@5``.
- **batch_size**: Batch size used during evaluation. Defaults to ``1024``.
- **strategy**: Evaluation strategy: ``full`` or ``sampled``. ``sampled`` is recommended for large datasets. Defaults to ``full``.
- **num_negatives**: Number of negative samples used in the ``sampled`` strategy.
- **seed**: Random seed used for reproducibility in sampling. Defaults to ``42``.
- **stat_significance**: Nested section defining statistical significance tests.
- **full_evaluation_on_report**: Whether to perform full evaluation each epoch. Defaults to ``False``.
- **max_metric_per_row**: Number of metrics logged per row. Defaults to ``4``.
- **beta**: Beta value for F1-score computation. Defaults to ``1.0``.
- **pop_ratio**: Fraction of transactions considered popular. Defaults to ``0.8``.
- **save_evaluation**: Whether to save evaluation results. Defaults to ``True``.
- **save_per_user**: Whether to save per-user evaluation results. Defaults to ``False``.

.. note::
    The Beta and Popularity ratio parameters are set to default values commonly used in typical experimental setups. Modifying these values may affect the results of certain metrics.

.. _configuration_stat_significance:

Statistical Significance
------------------------

The **stat_significance** nested section allows users to configure statistical tests for evaluating metric differences:

- **paired_t_test**: Enable the Paired t-test. Defaults to ``False``.
- **wilcoxon_test**: Enable the Wilcoxon signed-rank test. Defaults to ``False``.
- **kruskal_test**: Enable the Kruskal-Wallis H-test. Defaults to ``False``.
- **whitney_u_test**: Enable the Mann–Whitney U test. Defaults to ``False``.
- **corrections**: Nested section defining corrections for multiple hypothesis testing.

Corrections
-----------

The **corrections** section specifies methods to control family-wise error rate or false discovery rate:

- **bonferroni**: Apply Bonferroni correction. Defaults to ``False``.
- **holm_bonferroni**: Apply Holm-Bonferroni correction. Defaults to ``False``.
- **fdr**: Apply False Discovery Rate (FDR) correction. Defaults to ``False``.
- **alpha**: Significance level (α) for hypothesis testing. Defaults to ``0.05``.

Example Evaluation Configuration
--------------------------------

The following example evaluates the best model trained in the current iteration, using sampled evaluation and statistical tests:

.. code-block:: yaml

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
