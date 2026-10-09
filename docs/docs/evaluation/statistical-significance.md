# Statistical Significance

The evaluation module provides utilities for conducting **statistical significance testing** on computed evaluation metrics. These tests are performed on *pairs of models*, thus requiring at least two models to be included in the experiment.

For each model pair, significance tests are executed across all combinations of cutoff and metric. The purpose of these tests is to determine whether the observed differences in metric values between models are due to random variation or represent statistically meaningful improvements.

!!! note

    Detailed configuration options and further information on supported tests can be found in the configuration documentation.

## Supported Tests

Every test receives the two models' per-user values of one metric at one cutoff,
over the users both models have a value for. Since both models are scored on the
same users, the samples are paired.

- **Paired t-test** (paired): compares the mean of the per-user differences with zero.
- **Wilcoxon signed-rank test** (paired): its non-parametric counterpart, on the ranks
  of the per-user differences.
- **Kruskal-Wallis H-test** (unpaired): compares the two sets of per-user values as
  independent samples.
- **Mann-Whitney U test** (unpaired): likewise, as independent samples.

The two unpaired tests ignore that a user is scored by both models, so they answer a
weaker question and usually find fewer differences; WarpRec logs a note when one of
them is used. Prefer the Wilcoxon signed-rank test or the paired t-test for comparing
models.

## Reading the Results

Each enabled test writes one table, with a row per pair of models, metric and
cutoff: the statistic, the p-value and `Significance (α=…)`, which is `Significant`
when the p-value is below $\alpha$ and `Not significant` otherwise.

## Corrections for Multiple Testing

The `corrections` provides methods to control the **family-wise error rate (FWER)** or the **false discovery rate (FDR)** when multiple hypotheses are tested simultaneously:

- **Bonferroni correction** (FWER): a difference is significant when its p-value is
  below $\alpha / m$, for $m$ tests.
- **Holm-Bonferroni correction** (FWER): the p-values are sorted, the $i$-th smallest
  is compared with $\alpha / (m - i + 1)$, and testing stops at the first that fails;
  every difference from there on is not significant, whatever its p-value. It rejects
  at least as much as Bonferroni.
- **False Discovery Rate correction** (Benjamini-Hochberg): the p-values are sorted,
  the largest rank $k$ with $p_{(k)} < k\alpha / m$ is found, and the $k$ smallest are
  significant, including those that missed their own threshold.

Each correction adds its own `Significance (…)` column. The $m$ tests corrected for are
all the rows of the test's table, every pair of models, metric and cutoff together.
Holm-Bonferroni and FDR return the table sorted by p-value.

By enabling these tests and corrections, WarpRec allows users to assess not only the raw performance of models, but also the **robustness and reliability** of the observed differences.
