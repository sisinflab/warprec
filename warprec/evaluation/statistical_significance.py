# pylint: disable = too-few-public-methods
from typing import Any, Dict, Literal, Optional, Set, Tuple, cast
from abc import ABC, abstractmethod
from itertools import combinations

import numpy as np
import narwhals as nw
from narwhals.dataframe import DataFrame
from torch import Tensor
from scipy.stats import wilcoxon, ttest_rel, kruskal, mannwhitneyu

from warprec.utils.registry import stat_significance_registry
from warprec.utils.logger import logger

# What a significance column says about the difference between two models
SIGNIFICANT = "Significant"
NOT_SIGNIFICANT = "Not significant"


class StatisticalTest(ABC):
    """Abstract base class for statistical tests.
    This class defines the interface for statistical tests to be implemented.

    Attributes:
        PAIRED (bool): Whether the test uses the pairing of the two samples. Two
            models are scored on the same users, so a paired test compares each
            user with themself; an unpaired one treats the two sets of per-user
            scores as independent samples, and asks a weaker question.
    """

    PAIRED: bool = True

    @abstractmethod
    def compute(self, X: np.ndarray, Y: Optional[np.ndarray]) -> Tuple[float, float]:
        """Compute the statistical test.

        Args:
            X (np.ndarray): The first set of values.
            Y (Optional[np.ndarray]): The second set of values, if applicable.

        Returns:
            Tuple[float, float]: The statistic and p-value.
        """


@stat_significance_registry.register("wilcoxon_test")
class WilcoxonTest(StatisticalTest):
    """Wilcoxon signed-rank test implementation.
    This class implements the Wilcoxon signed-rank test for paired samples.
    """

    def compute(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Compute the Wilcoxon signed-rank test.

        Args:
            X (np.ndarray): The first set of values.
            Y (Optional[np.ndarray]): The second set of values, if applicable.

        Returns:
            Tuple[float, float]: The statistic and p-value.
        """

        stat, p = wilcoxon(X, Y)
        return stat, p


@stat_significance_registry.register("paired_t_test")
class PairedTTest(StatisticalTest):
    """Paired t-test implementation.
    This class implements the paired t-test
    for comparing two related samples.
    """

    def compute(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Compute the paired t-test.

        Args:
            X (np.ndarray): The first set of values.
            Y (Optional[np.ndarray]): The second set of values, if applicable.

        Returns:
            Tuple[float, float]: The t-statistic and p-value.
        """

        stat, p = ttest_rel(X, Y)
        return stat, p


@stat_significance_registry.register("kruskal_test")
class KruskalTest(StatisticalTest):
    """Kruskal-Wallis H test implementation.
    This class implements the Kruskal-Wallis H test for independent samples. It
    ignores that two models are scored on the same users; the Wilcoxon
    signed-rank test is its paired counterpart.
    """

    PAIRED = False

    def compute(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Compute the Kruskal-Wallis H test.

        Args:
            X (np.ndarray): The first set of values.
            Y (Optional[np.ndarray]): The second set of values, if applicable.

        Returns:
            Tuple[float, float]: The H statistic and p-value.
        """
        stat, p = kruskal(X, Y)
        return stat, p


@stat_significance_registry.register("whitney_u_test")
class WhitneyUTest(StatisticalTest):
    """Mann-Whitney U test implementation.
    This class implements the Mann-Whitney U test for independent samples. It
    ignores that two models are scored on the same users; the Wilcoxon
    signed-rank test is its paired counterpart.
    """

    PAIRED = False

    def compute(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Compute the Mann-Whitney U test.

        Args:
            X (np.ndarray): The first set of values.
            Y (Optional[np.ndarray]): The second set of values, if applicable.

        Returns:
            Tuple[float, float]: The U statistic and p-value.
        """
        stat, p = mannwhitneyu(X, Y)
        return stat, p


def apply_bonferroni_correction(
    results: DataFrame[Any], alpha: float = 0.05
) -> DataFrame[Any]:
    """Apply Bonferroni correction to p-values in the results DataFrame.

    Args:
        results (DataFrame[Any]): The DataFrame containing p-values.
        alpha (float): The significance level for the correction.

    Returns:
        DataFrame[Any]: The DataFrame with corrected significance values.
    """
    n_tests = results.select(nw.len()).item()
    corrected_alpha = alpha / n_tests

    return results.with_columns(
        nw.when(nw.col("p-value") < corrected_alpha)
        .then(nw.lit(SIGNIFICANT))
        .otherwise(nw.lit(NOT_SIGNIFICANT))
        .alias(f"Significance (Bonferroni α={corrected_alpha:.2e})")
    )


def apply_holm_bonferroni_correction(
    results: DataFrame[Any], alpha: float = 0.05
) -> DataFrame[Any]:
    """Apply Holm-Bonferroni correction to p-values in the results DataFrame.

    The table is returned sorted by p-value.

    Args:
        results (DataFrame[Any]): The DataFrame containing p-values.
        alpha (float): The significance level for the correction.

    Returns:
        DataFrame[Any]: The DataFrame with corrected significance values.
    """
    # Holm's step-down procedure: with the p-values sorted, the i-th smallest
    # of m is tested against alpha / (m - i + 1), and testing stops at the first
    # that fails. Every hypothesis from there on is kept, whatever its p-value.
    results = results.sort("p-value")
    p_values = results["p-value"].to_numpy()
    m = len(p_values)
    passes = p_values < alpha / (m - np.arange(m))
    rejected = np.logical_and.accumulate(passes) if m else passes
    return _with_verdicts(results, rejected, "Significance (Holm-Bonferroni)")


def apply_fdr_correction(
    results: DataFrame[Any], alpha: float = 0.05
) -> DataFrame[Any]:
    """Apply False Discovery Rate (FDR) correction to p-values in the results DataFrame.

    The Benjamini-Hochberg procedure. The table is returned sorted by p-value.

    Args:
        results (DataFrame[Any]): The DataFrame containing p-values.
        alpha (float): The significance level for the correction.

    Returns:
        DataFrame[Any]: The DataFrame with corrected significance values.
    """
    # Benjamini-Hochberg's step-up procedure: with the p-values sorted, find the
    # largest rank k whose p-value is below k * alpha / m, and reject every
    # hypothesis up to it, including those that missed their own threshold.
    results = results.sort("p-value")
    p_values = results["p-value"].to_numpy()
    m = len(p_values)
    passing = np.flatnonzero(p_values < alpha * np.arange(1, m + 1) / m)
    rejected = np.arange(m) <= passing[-1] if passing.size else np.zeros(m, bool)
    return _with_verdicts(results, rejected, "Significance (FDR)")


def _with_verdicts(
    results: DataFrame[Any], rejected: np.ndarray, column: str
) -> DataFrame[Any]:
    """Add one verdict per row of a table already in the order of the decisions.

    Args:
        results (DataFrame[Any]): The table, sorted as the decisions are.
        rejected (np.ndarray): Whether each row's null hypothesis is rejected.
        column (str): The name of the verdict column.

    Returns:
        DataFrame[Any]: The table with the verdict column added.
    """
    verdicts = np.where(rejected, SIGNIFICANT, NOT_SIGNIFICANT).tolist()
    return results.with_columns(
        nw.new_series(
            column, verdicts, nw.String, backend=nw.get_native_namespace(results)
        )
    )


def _note_unpaired(test_name: str, stat_test: StatisticalTest) -> None:
    """Say so when a test ignores that both models are scored on the same users.

    Args:
        test_name (str): The registered name of the test.
        stat_test (StatisticalTest): The test about to be run.
    """
    if not stat_test.PAIRED:
        logger.attention(
            f"{test_name} treats the two models' per-user scores as independent "
            "samples, although both are scored on the same users. The Wilcoxon "
            "signed-rank test and the paired t-test use that pairing."
        )


def compute_paired_statistical_test(
    results: Dict[str, Dict[int, Dict[str, float | Tensor]]],
    test_name: str,
    alpha: float = 0.05,
    bonferroni: bool = False,
    holm_bonferroni: bool = False,
    fdr: bool = False,
    backend: str = "polars",
) -> DataFrame[Any]:
    """Compute pairwise statistical significance tests on evaluation results.

    Args:
        results (Dict[str, Dict[int, Dict[str, float | Tensor]]]):
            Evaluation results structured as:
            {
                "model_name": {
                    "cutoff": {
                        "metric_name": value
                    }
                }
            }
        test_name (str): Name of the statistical test to use, e.g., "wilcoxon_test".
        alpha (float): Significance level for the statistical tests.
        bonferroni (bool): Whether to apply Bonferroni correction.
        holm_bonferroni (bool): Whether to apply Holm-Bonferroni correction.
        fdr (bool): Whether to apply False Discovery Rate correction.
        backend (str): The dataframe backend the results are built with, so that
            they match the rest of the experiment's frames.

    Returns:
        DataFrame[Any]: A DataFrame containing the results of the pairwise statistical tests.
    """
    # Initialize information for pairwise statistical test
    rows = []
    model_names = list(results.keys())
    stat_test: StatisticalTest = stat_significance_registry.get(test_name)
    _note_unpaired(test_name, stat_test)
    cutoff_values: Set[int] = set()

    # Gather all cutoff values
    for model in model_names:
        cutoff_values.update(results[model].keys())

    # Perform pairwise statistical tests
    for cutoff in sorted(cutoff_values):
        metric_names: Set[str] = set()
        for model in model_names:
            try:
                metric_names.update(results[model][cutoff].keys())
            except KeyError:
                continue

        for metric in sorted(metric_names):
            for model_a, model_b in combinations(
                model_names, 2
            ):  # Find all combinations
                try:
                    values_a = results[model_a][cutoff][metric]
                    values_b = results[model_b][cutoff][metric]

                    if isinstance(values_a, Tensor) and isinstance(values_b, Tensor):
                        # If the metric returns a Tensor, it was computed user-wise
                        # Convert to numpy arrays for statistical testing
                        # NOTE: float values come from metrics that cannot be computed user-wise
                        array_a = values_a.cpu().numpy()
                        array_b = values_b.cpu().numpy()

                        # Clean NaN values from arrays
                        # NOTE: NaN values can appear when users have no interactions
                        mask = ~np.isnan(array_a) & ~np.isnan(array_b)
                        array_a = array_a[mask]
                        array_b = array_b[mask]

                        # Safety check for minimum number of samples
                        if len(array_a) < 2 or len(array_b) < 2:
                            logger.attention(
                                f"Not enough valid samples for {model_a} vs {model_b} on {metric}"
                            )
                            continue

                        stat, p = stat_test.compute(array_a, array_b)
                        verdict = SIGNIFICANT if p < alpha else NOT_SIGNIFICANT

                        rows.append(
                            {
                                "Model A": model_a,
                                "Model B": model_b,
                                "Metric": metric,
                                "Cutoff": cutoff,
                                "Statistic": stat,
                                "p-value": p,
                                f"Significance (α={alpha})": verdict,
                            }
                        )
                except (KeyError, ValueError, AttributeError) as e:
                    logger.negative(
                        f"Error on {model_a} vs {model_b} | {metric} @ {cutoff}: {e}"
                    )

    # Convert to DataFrame
    data_dict = {k: [r[k] for r in rows] for k in rows[0].keys()}
    # GeneralConfig validates the backend, so it is one of the two supported
    # here. Narwhals types the argument as a Literal, which the configuration
    # cannot express, hence the cast.
    stat_test_df = nw.from_dict(
        data_dict, backend=cast(Literal["pandas", "polars"], backend)
    )

    # Apply corrections
    if bonferroni:
        stat_test_df = apply_bonferroni_correction(stat_test_df, alpha)
    if holm_bonferroni:
        stat_test_df = apply_holm_bonferroni_correction(stat_test_df, alpha)
    if fdr:
        stat_test_df = apply_fdr_correction(stat_test_df, alpha)

    return stat_test_df
