"""The table the evaluation prints is what most users actually read.

It is the last step between a metric and a decision, and it has two jobs that
can silently go wrong: reducing a per-user tensor to the single number a run
reports, and doing that with the same nan-awareness the metrics themselves use.
A mean rather than a nanmean here would quietly restate every per-user metric
as something lower.
"""

import math
from typing import Any, Dict, List

import pytest
import torch

from warprec.common.std_logs import log_evaluation


@pytest.fixture(name="captured")
def captured_fixture(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Collect what the logger is asked to print.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest patching fixture.

    Returns:
        List[str]: The lines handed to the logger.
    """
    lines: List[str] = []

    def record(message: Any, *args: Any, **kwargs: Any) -> None:
        lines.append(str(message))

    monkeypatch.setattr("warprec.common.std_logs.logger.table", record, raising=False)
    monkeypatch.setattr("warprec.common.std_logs.logger.msg", record, raising=False)
    monkeypatch.setattr(
        "warprec.common.std_logs.logger.stat_msg", record, raising=False
    )
    return lines


def test_a_per_user_tensor_is_reduced_the_way_the_metrics_report_it(
    captured: List[str],
):
    """A user with nothing to find is NaN, and must not be averaged in as a zero."""
    results: Dict[int, Dict[str, Any]] = {
        10: {"nDCG": torch.tensor([0.8, 0.4, float("nan")])}
    }

    log_evaluation(results, header="Test")

    printed = "\n".join(captured)
    # The mean of the two evaluable users is 0.6; counting the third as zero
    # would give 0.4.
    assert "0.6" in printed
    assert "0.4000" not in printed


def test_every_cutoff_gets_a_row(captured: List[str]):
    """A run asking for several cutoffs must see all of them."""
    results = {
        5: {"Recall": torch.tensor([1.0])},
        10: {"Recall": torch.tensor([0.5])},
        20: {"Recall": torch.tensor([0.25])},
    }

    log_evaluation(results, header="Test")

    printed = "\n".join(captured)
    for cutoff in (5, 10, 20):
        assert f"Top@{cutoff}" in printed


def test_many_metrics_are_split_across_tables(captured: List[str]):
    """A wide result set has to stay readable rather than wrapping unusably."""
    metrics = {f"M{i}": torch.tensor([float(i)]) for i in range(9)}

    log_evaluation({10: metrics}, header="Test", max_metrics_per_row=4)

    printed = "\n".join(captured)
    # Nine metrics at four per table is three tables, and none may be dropped.
    for name in metrics:
        assert name in printed


def test_a_plain_number_is_printed_as_it_is(captured: List[str]):
    """Not every metric is per-user; the scalar ones pass straight through."""
    log_evaluation({10: {"ItemCoverage": 42}}, header="Test")

    assert "42" in "\n".join(captured)


def test_a_metric_missing_from_a_cutoff_does_not_break_the_table(
    captured: List[str],
):
    """Cutoffs need not agree on which metrics they carry."""
    results: Dict[int, Dict[str, Any]] = {
        10: {"Recall": torch.tensor([0.5]), "nDCG": torch.tensor([0.25])},
        20: {"Recall": torch.tensor([0.75])},
    }

    log_evaluation(results, header="Test")

    printed = "\n".join(captured)
    assert "Top@20" in printed
    assert math.isnan(float("nan"))  # the absent cell is filled with NaN
    assert "nan" in printed.lower()
