.. _statistical_rigor:

############################
Statistical Rigor
############################

WarpRec automates statistical hypothesis testing and error correction to guarantee that observed performance differences between models are statistically robust, not artifacts of random variation.

.. contents:: On this page
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

-----

The Multiple Comparison Problem
================================

When comparing multiple recommendation models across several metrics and cutoffs, the probability of finding at least one spurious "significant" result grows rapidly. With :math:`m` independent tests at significance level :math:`\alpha`, the family-wise error rate (FWER) is:

.. math::

    \text{FWER} = 1 - (1 - \alpha)^m

For example, comparing 5 models on 3 metrics at 4 cutoffs yields :math:`\binom{5}{2} \times 3 \times 4 = 120` pairwise tests. At :math:`\alpha = 0.05`, the probability of at least one false positive is :math:`1 - 0.95^{120} \approx 99.8\%`.

WarpRec mitigates this by integrating **correction methods** directly into the evaluation pipeline. Statistical significance testing is supported in both the :ref:`pipeline-train` and the :ref:`pipeline-eval`.

-----

Supported Tests
===============

WarpRec provides four statistical tests for comparing model performance distributions. Tests are computed **per metric, per cutoff, per model pair**, using per-user metric vectors.

Paired Tests (Related Samples)
------------------------------

These tests compare two models evaluated on the **same set of users**.

**Paired t-test**

Tests whether the mean difference between two related samples is zero, assuming the differences are normally distributed.

.. math::

    t = \frac{\bar{d}}{s_d / \sqrt{n}}

where :math:`\bar{d}` is the mean of pairwise differences, :math:`s_d` is the standard deviation, and :math:`n` is the number of users.

**Wilcoxon signed-rank test**

A non-parametric alternative to the paired t-test. It does not assume normality, instead testing whether the median of pairwise differences is zero by ranking absolute differences.

Independent Tests
-----------------

**Mann-Whitney U test**

A non-parametric test for comparing two independent samples. Tests whether one distribution tends to produce larger values than the other.

**Kruskal-Wallis H test**

A non-parametric one-way ANOVA for comparing more than two groups. Tests whether the samples originate from the same distribution.

-----

Correction Methods
==================

WarpRec applies the following corrections to control for multiple comparisons:

Bonferroni Correction
---------------------

The simplest and most conservative correction. It divides the significance threshold by the total number of tests:

.. math::

    \alpha_{\text{corrected}} = \frac{\alpha}{m}

where :math:`m` is the number of pairwise comparisons. A test is significant if :math:`p < \alpha_{\text{corrected}}`.

Holm-Bonferroni Correction
--------------------------

A **step-down** procedure that is uniformly more powerful than Bonferroni while still controlling the FWER. The p-values are sorted in ascending order :math:`p_{(1)} \leq p_{(2)} \leq \dots \leq p_{(m)}`, and each is compared against a progressively relaxed threshold:

.. math::

    \alpha_i = \frac{\alpha}{m - i + 1}

Rejection proceeds sequentially: stop at the first :math:`p_{(i)} \geq \alpha_i` and reject all hypotheses before it.

False Discovery Rate (FDR) Correction
--------------------------------------

Controls the expected proportion of false positives among rejected hypotheses, rather than the probability of any false positive (FWER). The Benjamini-Hochberg procedure sorts p-values and compares:

.. math::

    p_{(i)} \leq \frac{i}{m} \cdot \alpha

This is less conservative than Bonferroni and more appropriate when a small fraction of false positives is tolerable.

-----

Configuration
=============

Statistical testing is configured in the ``evaluation`` section:

.. code-block:: yaml

    evaluation:
        top_k: [10, 20, 50]
        metrics: [nDCG, Precision, Recall, HitRate]
        stat_significance:
            wilcoxon_test: true
            corrections:
                bonferroni: true
                holm_bonferroni: true
                fdr: true
                alpha: 0.05

.. list-table:: Statistical Significance Configuration
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``wilcoxon_test``
     - ``false``
     - Enable the Wilcoxon signed-rank test.
   * - ``paired_ttest``
     - ``false``
     - Enable the paired t-test.
   * - ``kruskal_test``
     - ``false``
     - Enable the Kruskal-Wallis H test.
   * - ``whitney_u_test``
     - ``false``
     - Enable the Mann-Whitney U test.
   * - ``corrections.bonferroni``
     - ``false``
     - Apply Bonferroni correction.
   * - ``corrections.holm_bonferroni``
     - ``false``
     - Apply Holm-Bonferroni correction.
   * - ``corrections.fdr``
     - ``false``
     - Apply FDR (Benjamini-Hochberg) correction.
   * - ``corrections.alpha``
     - ``0.05``
     - Significance level for all tests and corrections.

.. important::

    Statistical tests are performed on **pairs of models**, so at least **two models** must be included in the experiment.

-----

Output Format
=============

For each enabled test, WarpRec produces a results table with the following columns:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Description
   * - Model A
     - First model in the comparison.
   * - Model B
     - Second model in the comparison.
   * - Metric
     - The evaluation metric being compared.
   * - Cutoff
     - The top-K cutoff value.
   * - Statistic
     - The test statistic value.
   * - p-value
     - The raw p-value from the test.
   * - Significance (Bonferroni)
     - ``Accepted`` or ``Rejected`` (if enabled).
   * - Significance (Holm-Bonferroni)
     - ``Accepted`` or ``Rejected`` (if enabled).
   * - Significance (FDR)
     - ``Accepted`` or ``Rejected`` (if enabled).

Results are saved to the experiment directory via the Writer module.

.. seealso::

   :ref:`api_evaluation` for the ``compute_paired_statistical_test`` function API reference.

-----

Practical Example
=================

The following configuration compares three models with full statistical testing:

.. code-block:: yaml

    reader:
        loading_strategy: dataset
        reading_method: local
        local_path: data/movielens.tsv
        rating_type: implicit
    writer:
        dataset_name: StatisticalTest
        writing_method: local
        local_experiment_path: experiments/stat/
    splitter:
        test_splitting:
            strategy: temporal_holdout
            ratio: 0.1
    models:
        EASE:
            l2: 10
        ItemKNN:
            k: 100
            similarity: cosine
        BPR:
            embedding_size: 64
            reg_weight: 0.001
            batch_size: 2048
            epochs: 100
            learning_rate: 0.001
    evaluation:
        top_k: [10, 20]
        metrics: [nDCG, Precision, Recall]
        stat_significance:
            wilcoxon_test: true
            paired_ttest: true
            corrections:
                bonferroni: true
                fdr: true
                alpha: 0.05

This generates :math:`\binom{3}{2} \times 3 \times 2 = 18` pairwise comparisons per test, with both Bonferroni and FDR corrections applied.
