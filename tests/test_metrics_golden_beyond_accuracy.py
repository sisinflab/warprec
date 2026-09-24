"""Bias, diversity, novelty and coverage metrics, pinned against hand-computed values.

These are the numbers a paper reports when it claims a model is fair, diverse or
popularity-unbiased, and they are the easiest in the framework to get quietly
wrong: unlike an accuracy metric there is no intuition for what the right answer
looks like, so a wrong one is simply believed. Every value below is derived in
the comments from the fixture, not captured from a run.

The fixture is the one `test_metrics_golden` uses, so the rankings are already
established there:

    user 0  trained {1, 2}   relevant {0, 3}   top-3 after masking: 4, 3, 0
    user 1  trained {0, 3}   relevant {4}      top-3 after masking: 1, 4, 2
    user 2  trained {4, 0}   no test items     excluded, scores NaN

Training popularity is item0: 2, items 1-4: 1 each, six interactions in all.
"""

import math
from typing import Dict

import pandas as pd
import pytest
import torch

from test_metrics_golden import _FixedScoreRecommender, TEST_ROWS, TRAIN_ROWS

from warprec.data.dataset import Dataset
from warprec.evaluation import Evaluator

# The long tail is what is left after the most popular items covering 80% of the
# interactions are removed. Sorted counts are [2, 1, 1, 1, 1] over six
# interactions, so the cumulative total first passes 4.8 at the fourth item:
# items 0-3 are the short head and item 4 alone is the tail.
#
#   ACLT  long-tail items per list:  user 0 has {4} -> 1,  user 1 has {4} -> 1
#   APLT  as a share of the list:    1/3 and 1/3
#   ARP   mean training popularity:  (1 + 1 + 2)/3 and (1 + 1 + 1)/3
EXPECTED_BIAS: Dict[str, float] = {
    "ACLT": (1 + 1) / 2,
    "APLT": (1 / 3 + 1 / 3) / 2,
    "ARP": (4 / 3 + 1) / 2,
}

# Across the two evaluable users the six recommended slots fall on item 4 twice
# and items 0, 1, 2, 3 once each, so the distribution is p = [1, 1, 1, 1, 2] / 6.
_COUNTS = [1, 1, 1, 1, 2]
_TOTAL = sum(_COUNTS)
_SHARES = [count / _TOTAL for count in _COUNTS]

# Gini over that distribution, sorted ascending, with n = 5 recommended items:
#   G = sum_j (2j - n - 1) * p_j / (n - 1),  j counted from one
_GINI = sum(
    (2 * (position + 1) - len(_COUNTS) - 1) * share
    for position, share in enumerate(_SHARES)
) / (len(_COUNTS) - 1)

# Shannon entropy of the same distribution, in nats.
_SHANNON = -sum(share * math.log(share) for share in _SHARES)

EXPECTED_DIVERSITY: Dict[str, float] = {"Gini": _GINI, "ShannonEntropy": _SHANNON}

# The novelty metrics discount by rank and count only the relevant hits, then
# divide by the discount the whole list could have earned.
_DISCOUNTS = [1 / math.log2(rank + 1) for rank in (1, 2, 3)]
_NORM = sum(_DISCOUNTS)

# EPC reads popularity as a share of the users, EFD as a share of the
# interactions, which is the usual pairing for these two.
#   user 0 hits item 3 at rank 2 and item 0 at rank 3
#   user 1 hits item 4 at rank 2
_U0_EPC = (_DISCOUNTS[1] * (1 - 1 / 3) + _DISCOUNTS[2] * (1 - 2 / 3)) / _NORM
_U1_EPC = (_DISCOUNTS[1] * (1 - 1 / 3)) / _NORM
_U0_EFD = (
    _DISCOUNTS[1] * -math.log2(1 / 6) + _DISCOUNTS[2] * -math.log2(2 / 6)
) / _NORM
_U1_EFD = (_DISCOUNTS[1] * -math.log2(1 / 6)) / _NORM

EXPECTED_NOVELTY: Dict[str, float] = {
    "EPC": (_U0_EPC + _U1_EPC) / 2,
    "EFD": (_U0_EFD + _U1_EFD) / 2,
}

# Every slot of both lists is filled, five distinct items are touched across
# them, and two users could be evaluated at all.
EXPECTED_COVERAGE: Dict[str, float] = {
    "NumRetrieved": 3.0,
    "ItemCoverage": 5,
    "UserCoverage": 2,
}

# MAR averages the recall reached at each hit.
#   user 0: hits at ranks 2 and 3 of two relevant -> (1/2 + 2/2) / min(2, 3)
#   user 1: hit at rank 2 of one relevant         -> (1/1) / min(1, 3)
EXPECTED_ACCURACY: Dict[str, float] = {
    "MAR": ((0.5 + 1.0) / 2 + 1.0) / 2,
    # F1 is the harmonic mean of the precision and recall of each evaluable user.
    #   user 0: P = 2/3, R = 1 -> 0.8      user 1: P = 1/3, R = 1 -> 0.5
    "F1": (0.8 + 0.5) / 2,
}

ALL_EXPECTED = {
    **EXPECTED_BIAS,
    **EXPECTED_DIVERSITY,
    **EXPECTED_NOVELTY,
    **EXPECTED_COVERAGE,
    **EXPECTED_ACCURACY,
}


@pytest.fixture(name="fixture_dataset", scope="module")
def fixture_dataset_fixture() -> Dataset:
    """The three-user fixture the expected values are derived from.

    Returns:
        Dataset: The dataset under test.
    """
    columns = ["user_id", "item_id", "rating"]
    return Dataset(
        train_data=pd.DataFrame(TRAIN_ROWS, columns=columns),
        eval_data=pd.DataFrame(TEST_ROWS, columns=columns),
        rating_type="explicit",
        rating_label="rating",
        batch_size=8,
    )


def measured(metric_name: str, dataset: Dataset) -> float:
    """Run one metric through the real evaluator and report what a user sees.

    Args:
        metric_name (str): The registered metric name.
        dataset (Dataset): The fixture.

    Returns:
        float: The reported value, which is the nanmean for per-user metrics.
    """
    evaluator = Evaluator([metric_name], [3], train_set=dataset.train_set.get_sparse())
    evaluator.evaluate(
        model=_FixedScoreRecommender({}, dataset.info()),
        dataloader=dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=dataset,
    )
    value = evaluator.compute_results()[3][metric_name]

    if torch.is_tensor(value) and value.numel() > 1:
        return float(value.nanmean())
    return float(value)


@pytest.mark.parametrize("metric_name,expected", sorted(ALL_EXPECTED.items()))
def test_metric_matches_hand_computed_value(
    metric_name: str, expected: float, fixture_dataset: Dataset
):
    """Each metric reproduces the value derived in the header."""
    got = measured(metric_name, fixture_dataset)

    assert got == pytest.approx(expected, abs=1e-5), (
        f"{metric_name}@3 is {got}, expected {expected}"
    )


def test_a_user_with_nothing_to_find_is_left_out_of_the_mean(
    fixture_dataset: Dataset,
):
    """User 2 has no test items, so it must not be averaged in as a zero.

    F1 used to force its NaN to zero before the mean was taken, which pulled the
    reported score down by however many such users a split happened to contain:
    0.433 here instead of 0.65, on a fixture where a third of the users are in
    that position.
    """
    evaluator = Evaluator(
        ["F1", "Precision", "Recall"],
        [3],
        train_set=fixture_dataset.train_set.get_sparse(),
    )
    evaluator.evaluate(
        model=_FixedScoreRecommender({}, fixture_dataset.info()),
        dataloader=fixture_dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=fixture_dataset,
    )
    results = evaluator.compute_results()[3]

    # Whatever the other two do about the third user, F1 has to do the same.
    for name in ("Precision", "Recall", "F1"):
        assert torch.isnan(results[name][2]), f"{name} counted a user it cannot score"

    assert float(results["F1"].nanmean()) == pytest.approx(0.65, abs=1e-5)
