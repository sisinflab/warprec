"""Behavioural tests for the filters.

A filter only ever removes transactions, so two promises hold for all of them:
the output is a subset of the input, and its columns are untouched. On top of
that each filter has to enforce the condition it is named after.
"""

from typing import Set, Tuple

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

from warprec.utils.registry import filter_registry

ALL_FILTERS = {
    "MinRating": {"min_rating": 3.0},
    "UserAverage": {},
    "ItemAverage": {},
    "UserMin": {"min_interactions": 3},
    "UserMax": {"max_interactions": 4},
    "ItemMin": {"min_interactions": 3},
    "ItemMax": {"max_interactions": 4},
    "IterativeKCore": {"min_interactions": 3},
    "NRoundsKCore": {"rounds": 2, "min_interactions": 3},
    "UserHeadN": {"num_interactions": 3},
    "UserTailN": {"num_interactions": 3},
    "DropUser": {"user_ids_to_filter": [0, 1]},
    "DropItem": {"item_ids_to_filter": [0, 1]},
}


@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    """Interactions with an uneven spread, so every filter has work to do.

    Returns:
        pd.DataFrame: The generated transactions.
    """
    rng = np.random.default_rng(23)
    rows = []
    for user in range(20):
        # A user's history length varies from 1 to 10, so the count filters bite.
        for n, item in enumerate(rng.choice(30, 1 + user % 10, replace=False)):
            rows.append((user, int(item), float(rng.integers(1, 6)), user * 100 + n))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def apply(name: str, frame: pd.DataFrame, **params) -> pd.DataFrame:
    """Run one registered filter over a frame.

    Args:
        name (str): The registered filter name.
        frame (pd.DataFrame): The transactions to filter.
        **params: The filter's own parameters.

    Returns:
        pd.DataFrame: The filtered transactions.
    """
    out = filter_registry.get(name, **params)(nw.from_native(frame))
    return out.to_native() if hasattr(out, "to_native") else out


def pairs(frame: pd.DataFrame) -> Set[Tuple[int, int]]:
    """The (user, item) pairs a frame holds.

    Args:
        frame (pd.DataFrame): The frame to read.

    Returns:
        Set[Tuple[int, int]]: The pairs.
    """
    return set(map(tuple, frame[["user_id", "item_id"]].to_numpy().tolist()))


@pytest.mark.parametrize("name,params", sorted(ALL_FILTERS.items()))
def test_a_filter_only_removes(transactions: pd.DataFrame, name: str, params: dict):
    """Every filter returns a subset of its input, with the same columns."""
    out = apply(name, transactions, **params)

    assert list(out.columns) == list(transactions.columns)
    assert pairs(out) <= pairs(transactions), f"{name} invented a transaction"


def test_min_rating_drops_everything_below_the_threshold(transactions: pd.DataFrame):
    """MinRating keeps exactly the rows at or above the threshold."""
    out = apply("MinRating", transactions, min_rating=3.0)

    assert (out["rating"] >= 3.0).all()
    assert len(out) == int((transactions["rating"] >= 3.0).sum())


@pytest.mark.parametrize(
    "name,column", [("UserMin", "user_id"), ("ItemMin", "item_id")]
)
def test_min_count_filters_enforce_their_bound(
    transactions: pd.DataFrame, name: str, column: str
):
    """Nothing under the bound survives, and nothing over it is dropped."""
    out = apply(name, transactions, min_interactions=3)

    counts = out[column].value_counts()
    assert (counts >= 3).all()

    kept = set(transactions[column].value_counts().loc[lambda c: c >= 3].index)
    assert set(counts.index) == kept


@pytest.mark.parametrize(
    "name,column", [("UserMax", "user_id"), ("ItemMax", "item_id")]
)
def test_max_count_filters_enforce_their_bound(
    transactions: pd.DataFrame, name: str, column: str
):
    """No entity is left holding more than the cap."""
    out = apply(name, transactions, max_interactions=4)

    assert (out[column].value_counts() <= 4).all()


def test_iterative_k_core_reaches_a_fixpoint(transactions: pd.DataFrame):
    """A k-core is a fixpoint: running it again changes nothing."""
    once = apply("IterativeKCore", transactions, min_interactions=3)
    twice = apply("IterativeKCore", once, min_interactions=3)

    assert pairs(once) == pairs(twice)

    # And it is a genuine k-core: both sides satisfy the bound at once.
    assert (once["user_id"].value_counts() >= 3).all()
    assert (once["item_id"].value_counts() >= 3).all()


def test_n_rounds_k_core_is_bounded_by_the_iterative_one(transactions: pd.DataFrame):
    """A fixed number of rounds can only remove as much as running to a fixpoint."""
    rounds = apply("NRoundsKCore", transactions, rounds=2, min_interactions=3)
    fixpoint = apply("IterativeKCore", transactions, min_interactions=3)

    assert pairs(fixpoint) <= pairs(rounds)


@pytest.mark.parametrize(
    "name,take",
    [("UserHeadN", lambda g: g.head(3)), ("UserTailN", lambda g: g.tail(3))],
)
def test_head_and_tail_take_the_right_end_of_a_history(
    transactions: pd.DataFrame, name: str, take
):
    """Head takes a user's earliest interactions, tail their latest."""
    out = apply(name, transactions, num_interactions=3)

    expected = (
        transactions.sort_values(["user_id", "timestamp"])
        .groupby("user_id", group_keys=False)
        .apply(take)
    )
    assert pairs(out) == pairs(expected)


@pytest.mark.parametrize(
    "name,column,key",
    [
        ("DropUser", "user_id", "user_ids_to_filter"),
        ("DropItem", "item_id", "item_ids_to_filter"),
    ],
)
def test_drop_removes_exactly_the_named_ids(
    transactions: pd.DataFrame, name: str, column: str, key: str
):
    """Dropping ids removes those and only those."""
    out = apply(name, transactions, **{key: [0, 1]})

    assert not set(out[column]) & {0, 1}
    assert set(out[column]) == set(transactions[column]) - {0, 1}
