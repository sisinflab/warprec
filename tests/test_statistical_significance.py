"""Behavioural tests for the significance tests and their multiple-testing corrections.

The corrections are the part of a comparison a reader cannot check by eye, so each is
held to its textbook definition: Holm-Bonferroni stops at the first hypothesis it
cannot reject, Benjamini-Hochberg rejects every hypothesis up to the largest rank that
passes. The tables say "Significant" or "Not significant" for the difference itself.
"""

from typing import Any, List

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from scipy.stats import false_discovery_control

from warprec.evaluation.statistical_significance import (
    apply_bonferroni_correction,
    apply_fdr_correction,
    apply_holm_bonferroni_correction,
    compute_paired_statistical_test,
)
from warprec.utils.registry import stat_significance_registry

BACKENDS = ["pandas", "polars"]


def frame(p_values: List[float], backend: str) -> Any:
    """A results table holding only the p-values a correction reads.

    Args:
        p_values (List[float]): The p-values, in the order the table lists them.
        backend (str): 'pandas' or 'polars'.

    Returns:
        Any: A Narwhals frame with a 'p-value' and a 'Test' column.
    """
    data = {"Test": list(range(len(p_values))), "p-value": p_values}
    native = pd.DataFrame(data) if backend == "pandas" else pl.DataFrame(data)
    return nw.from_native(native)


def significant(result: Any, column: str) -> List[bool]:
    """Read a correction's verdicts back in the order the tests were listed.

    Args:
        result (Any): The corrected table.
        column (str): The column the correction wrote.

    Returns:
        List[bool]: Whether each test, by its original position, is significant.
    """
    rows = result.sort("Test").to_native()
    labels = list(rows[column])
    assert set(labels) <= {"Significant", "Not significant"}
    return [label == "Significant" for label in labels]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "p_values, expected",
    [
        # The second would pass alpha/1 on its own, but Holm stops at the first
        # hypothesis it cannot reject: 0.04 >= 0.05 / 2.
        ([0.04, 0.045], [False, False]),
        ([0.001, 0.02], [True, True]),
        # Sorted 0.01 < 0.05/3 passes, 0.03 >= 0.05/2 fails, so 0.04 is not
        # rejected even though it is below 0.05/1.
        ([0.01, 0.04, 0.03], [True, False, False]),
    ],
)
def test_holm_stops_at_the_first_hypothesis_it_cannot_reject(
    backend: str, p_values: List[float], expected: List[bool]
):
    result = apply_holm_bonferroni_correction(frame(p_values, backend), alpha=0.05)
    assert significant(result, "Significance (Holm-Bonferroni)") == expected


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "p_values, expected",
    [
        # 0.04 <= 2 * 0.05 / 2, so both ranks up to 2 are rejected.
        ([0.03, 0.04], [True, True]),
        # Only the smallest passes its own threshold.
        ([0.01, 0.04, 0.045, 0.2], [True, False, False, False]),
        # 0.02 misses 0.05/3 but the largest passing rank is 3, so all three go.
        ([0.02, 0.03, 0.04], [True, True, True]),
    ],
)
def test_benjamini_hochberg_rejects_up_to_the_largest_passing_rank(
    backend: str, p_values: List[float], expected: List[bool]
):
    result = apply_fdr_correction(frame(p_values, backend), alpha=0.05)
    assert significant(result, "Significance (FDR)") == expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_benjamini_hochberg_agrees_with_scipy(backend: str):
    rng = np.random.default_rng(0)
    for _ in range(200):
        p_values = np.concatenate(
            [rng.uniform(0, 0.02, rng.integers(0, 6)), rng.uniform(0, 1, 10)]
        ).tolist()
        reference = (false_discovery_control(p_values) < 0.05).tolist()
        result = apply_fdr_correction(frame(p_values, backend), alpha=0.05)
        assert significant(result, "Significance (FDR)") == reference


@pytest.mark.parametrize("backend", BACKENDS)
def test_bonferroni_divides_alpha_by_the_number_of_tests(backend: str):
    result = apply_bonferroni_correction(frame([0.01, 0.02, 0.2], backend), 0.05)
    column = next(c for c in result.columns if c.startswith("Significance (Bonferroni"))
    assert significant(result, column) == [True, False, False]


def test_the_table_says_whether_each_difference_is_significant():
    results = {
        "A": {10: {"nDCG": torch.linspace(0.5, 1.0, 50)}},
        "B": {10: {"nDCG": torch.linspace(0.0, 0.5, 50)}},
        "C": {10: {"nDCG": torch.linspace(0.0, 0.5, 50) + 1e-3}},
    }
    table = compute_paired_statistical_test(results, "paired_t_test", alpha=0.05)
    verdicts = dict(
        zip(
            zip(table["Model A"].to_list(), table["Model B"].to_list()),
            table["Significance (α=0.05)"].to_list(),
        )
    )
    assert verdicts[("A", "B")] == "Significant"
    assert set(verdicts.values()) <= {"Significant", "Not significant"}


@pytest.mark.parametrize(
    "name, paired",
    [
        ("wilcoxon_test", True),
        ("paired_t_test", True),
        ("kruskal_test", False),
        ("whitney_u_test", False),
    ],
)
def test_every_test_says_whether_it_uses_the_pairing(name: str, paired: bool):
    assert stat_significance_registry.get_class(name).PAIRED is paired
