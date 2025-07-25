from typing import Dict, Tuple, Optional, Set
from abc import ABC, abstractmethod
from itertools import combinations

import numpy as np
from pandas import DataFrame
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


def compute_paired_statistical_test(
    results: Dict[str, Dict[str, Dict[int, Dict[str, float | Tensor]]]], test_name: str
) -> DataFrame:
    """Compute pairwise statistical significance tests on evaluation results.

    Args:
        results (Dict[str, Dict[str, Dict[int, Dict[str, float | Tensor]]]]):
            Evaluation results structured as:
            {
                "model_name": {
                    "set_name": {
                        "cutoff": {
                            "metric_name": value
                        }
                    }
                }
            }
        test_name (str): Name of the statistical test to use, e.g., "wilcoxon_test".

    Returns:
        DataFrame: A DataFrame containing the results of the pairwise statistical tests.
    """
    # Initialize information for pairwise statistical test
    rows = []
    model_names = list(results.keys())
    set_names: Set[str] = set()
    for model_data in results.values():
        set_names.update(model_data.keys())
    stat_test: StatisticalTest = stat_significance_registry.get(test_name)

    # Compute on all sets
    for set_name in set_names:
        cutoff_values: Set[int] = set()
        for model in model_names:
            if set_name in results[model]:
                cutoff_values.update(results[model][set_name].keys())

        for cutoff in sorted(cutoff_values):
            metric_names: Set[str] = set()
            for model in model_names:
                try:
                    metric_names.update(results[model][set_name][cutoff].keys())
                except KeyError:
                    continue

            for metric in sorted(metric_names):
                for model_a, model_b in combinations(
                    model_names, 2
                ):  # Find all combinations
                    try:
                        values_a = results[model_a][set_name][cutoff][metric]
                        values_b = results[model_b][set_name][cutoff][metric]

                        if isinstance(values_a, Tensor) and isinstance(
                            values_b, Tensor
                        ):
                            # If the metric returns a Tensor, it was computed user-wise
                            # Convert to numpy arrays for statistical testing
                            # NOTE: float values come from metrics that cannot be computed user-wise
                            array_a = values_a.cpu().numpy()
                            array_b = values_b.cpu().numpy()

                            stat, p = stat_test.compute(array_a, array_b)

                            rows.append(
                                {
                                    "Model A": model_a,
                                    "Model B": model_b,
                                    "Set": set_name,
                                    "Cutoff": cutoff,
                                    "Metric": metric,
                                    "Statistic": stat,
                                    "p-value": p,
                                }
                            )
                    except Exception as e:
                        logger.negative(
                            f"Error on {model_a} vs {model_b} | {set_name} @ {cutoff} - {metric}: {e}"
                        )

    return DataFrame(rows)
