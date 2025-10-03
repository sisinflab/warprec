########################
Statistical Significance
########################

The evaluation module provides utilities for conducting **statistical significance testing** on computed evaluation metrics.
These tests are performed on *pairs of models*, thus requiring at least two models to be included in the experiment.

For each model pair, significance tests are executed across all combinations of cutoff and metric.
The purpose of these tests is to determine whether the observed differences in metric values between models are due to random variation or represent statistically meaningful improvements.

.. note::
   Detailed configuration options and further information on supported tests can be found in the :ref:`configuration documentation <configuration_stat_significance>`.

Supported Tests
---------------

Currently supported tests are:

- **Paired t-test**: Compares the means of two related samples.
- **Wilcoxon signed-rank test**: A non-parametric alternative to the paired t-test.
- **Kruskal–Wallis H-test**: A non-parametric test for comparing more than two groups.
- **Mann–Whitney U test**: Non-parametric test for independent samples.

Corrections for Multiple Testing
--------------------------------

The ``corrections`` provides methods to control the **family-wise error rate (FWER)** or the **false discovery rate (FDR)** when multiple hypotheses are tested simultaneously:

- **Bonferroni correction**: Adjusts the significance threshold by dividing :math:`\alpha` by the number of tests.
- **Holm–Bonferroni correction**: Sequentially rejects null hypotheses while controlling FWER.
- **False Discovery Rate (FDR) correction**: Controls the expected proportion of false positives among rejected hypotheses.

By enabling these tests and corrections, WarpRec allows users to assess not only the raw performance of models, but also the **robustness and reliability** of the observed differences.
