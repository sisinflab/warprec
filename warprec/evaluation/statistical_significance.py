# pylint: disable = too-few-public-methods
from typing import Dict, Tuple, Optional, Set, Any
from abc import ABC, abstractmethod
from itertools import combinations

import numpy as np
import narwhals as nw
from narwhals.dataframe import DataFrame
from torch import Tensor
from scipy.stats import wilcoxon, ttest_rel, kruskal, mannwhitneyu

from warprec.utils.registry import stat_significance_registry
from warprec.utils.logger import logger


class StatisticalTest(ABC):
    """Abstract base class for statistical tests.
    This class defines the interface for statistical tests to be implemented.
    """

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
    This class implements the Kruskal-Wallis H test for independent samples.
    """

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
    This class implements the Mann-Whitney U test for independent samples.
    """

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
        .then(nw.lit("Accepted"))
        .otherwise(nw.lit("Rejected"))
        .alias(f"Significance (Bonferroni α={corrected_alpha:.2e})")
    )


def apply_holm_bonferroni_correction(
    results: DataFrame[Any], alpha: float = 0.05
) -> DataFrame[Any]:
    """Apply Holm-Bonferroni correction to p-values in the results DataFrame.

    Args:
        results (DataFrame[Any]): The DataFrame containing p-values.
        alpha (float): The significance level for the correction.

    Returns:
        DataFrame[Any]: The DataFrame with corrected significance values.
    """
    results = results.sort("p-value")

    # Create a rank column (1-based)
    # We use 'ordinal' rank to get a simple 1..N sequence matching the sort
    results = results.with_columns(
        nw.col("p-value").rank(method="ordinal").alias("Holm-Bonferroni Rank")
    )

    # Apply logic
    results = results.with_columns(
        nw.when(
            nw.col("p-value")
            < (alpha / (nw.len() - nw.col("Holm-Bonferroni Rank") + 1))
        )
        .then(nw.lit("Accepted"))
        .otherwise(nw.lit("Rejected"))
        .alias("Significance (Holm-Bonferroni)")
    )

    return results.drop("Holm-Bonferroni Rank")


def apply_fdr_correction(
    results: DataFrame[Any], alpha: float = 0.05
) -> DataFrame[Any]:
    """Apply False Discovery Rate (FDR) correction to p-values in the results DataFrame.

    Args:
        results (DataFrame[Any]): The DataFrame containing p-values.
        alpha (float): The significance level for the correction.

    Returns:
        DataFrame[Any]: The DataFrame with corrected significance values.
    """
    results = results.sort("p-value")

    results = results.with_columns(
        nw.col("p-value").rank(method="ordinal").alias("FDR Rank")
    )

    # Calculate Threshold
    # Threshold = (Rank / Total) * Alpha
    results = results.with_columns(
        (alpha * nw.col("FDR Rank") / nw.len()).alias("FDR Threshold")
    )

    results = results.with_columns(
        nw.when(nw.col("p-value") < nw.col("FDR Threshold"))
        .then(nw.lit("Accepted"))
        .otherwise(nw.lit("Rejected"))
        .alias("Significance (FDR)")
    )

    return results.drop(["FDR Rank", "FDR Threshold"])


def compute_paired_statistical_test(
    results: Dict[str, Dict[int, Dict[str, float | Tensor]]],
    test_name: str,
    alpha: float = 0.05,
    bonferroni: bool = False,
    holm_bonferroni: bool = False,
    fdr: bool = False,
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

    Returns:
        DataFrame[Any]: A DataFrame containing the results of the pairwise statistical tests.
    """
    # Initialize information for pairwise statistical test
    rows = []
    model_names = list(results.keys())
    stat_test: StatisticalTest = stat_significance_registry.get(test_name)
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
                        accepted = "Accepted" if p < alpha else "Rejected"

                        rows.append(
                            {
                                "Model A": model_a,
                                "Model B": model_b,
                                "Metric": metric,
                                "Cutoff": cutoff,
                                "Statistic": stat,
                                "p-value": p,
                                f"Significance (α={alpha})": accepted,
                            }
                        )
                except (KeyError, ValueError, AttributeError) as e:
                    logger.negative(
                        f"Error on {model_a} vs {model_b} | {metric} @ {cutoff}: {e}"
                    )

    # Convert to DataFrame
    data_dict = {k: [r[k] for r in rows] for k in rows[0].keys()}
    stat_test_df = nw.from_dict(data_dict)

    # Apply corrections
    if bonferroni:
        stat_test_df = apply_bonferroni_correction(stat_test_df, alpha)
    if holm_bonferroni:
        stat_test_df = apply_holm_bonferroni_correction(stat_test_df, alpha)
    if fdr:
        stat_test_df = apply_fdr_correction(stat_test_df, alpha)

    return stat_test_df
