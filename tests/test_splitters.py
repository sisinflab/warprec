"""Behavioural tests for the splitting strategies.

Every strategy has to hold the same contract whatever it does internally: it
may not invent a transaction, it may not leave one in both halves, and a seeded
strategy has to land on the same split twice. The strategy-specific promises
are checked on top of that.
"""

from typing import Any, Set, Tuple

import numpy as np
import pandas as pd
import pytest

from warprec.data.schema import ColumnLabels, SplitSpec
from warprec.data.splitting import Splitter

HOLDOUTS = ["random_holdout", "temporal_holdout"]
LEAVE_K_OUT = ["random_leave_k_out", "temporal_leave_k_out"]


@pytest.fixture(scope="module")
def transactions() -> pd.DataFrame:
    """A frame dense enough that every user survives any split.

    Returns:
        pd.DataFrame: The generated transactions.
    """
    rng = np.random.default_rng(17)
    n_users, n_items, per_user = 40, 60, 12
    users = np.repeat(np.arange(n_users), per_user)
    items = np.concatenate(
        [rng.choice(n_items, per_user, replace=False) for _ in range(n_users)]
    )
    return pd.DataFrame(
        {
            "user_id": users,
            "item_id": items,
            "rating": rng.integers(1, 6, users.size).astype(float),
            # Distinct per row, so a temporal split has an unambiguous order.
            "timestamp": np.arange(users.size) * 10,
        }
    )


def pairs(frame: Any) -> Set[Tuple[int, int]]:
    """The (user, item) pairs a split half holds.

    Args:
        frame (Any): The frame to read, in any backend narwhals accepts.

    Returns:
        Set[Tuple[int, int]]: The pairs.
    """
    native = frame.to_native() if hasattr(frame, "to_native") else frame
    return set(map(tuple, native[["user_id", "item_id"]].to_numpy().tolist()))


def split(data: pd.DataFrame, **kwargs: Any) -> Any:
    """Run one test split with WarpRec's default column names.

    Args:
        data (pd.DataFrame): The frame to split.
        **kwargs (Any): The fields of the test SplitSpec.

    Returns:
        Any: The train set, the validation set and the test set.
    """
    return Splitter().split_transaction(
        data, labels=ColumnLabels(), test=SplitSpec(**kwargs)
    )


@pytest.mark.parametrize(
    "spec",
    [{"strategy": s, "ratio": 0.2} for s in HOLDOUTS]
    + [{"strategy": s, "k": 2} for s in LEAVE_K_OUT],
)
def test_split_partitions_the_transactions(transactions: pd.DataFrame, spec: dict):
    """Train and test must be disjoint, and together lose nothing."""
    train, _, test = split(transactions, **spec)

    train_pairs, test_pairs = pairs(train), pairs(test)
    assert train_pairs & test_pairs == set(), "a pair is in both halves"

    # The test set is filtered down to users and items the train set knows, so
    # it may shrink; nothing may appear that was not in the input.
    assert train_pairs | test_pairs <= pairs(transactions)
    assert test_pairs, "the test set came back empty"


@pytest.mark.parametrize("strategy", LEAVE_K_OUT)
def test_leave_k_out_holds_out_exactly_k(transactions: pd.DataFrame, strategy: str):
    """A leave-k-out split gives every user exactly k test transactions."""
    _, _, test = split(transactions, strategy=strategy, k=2)

    counts = pd.Series([u for u, _ in pairs(test)]).value_counts()
    assert set(counts.unique()) == {2}, f"users got {sorted(counts.unique())} rows"


def test_temporal_holdout_keeps_the_test_set_in_the_future(transactions: pd.DataFrame):
    """A temporal split may not leak a later interaction into training."""
    train, _, test = split(transactions, strategy="temporal_holdout", ratio=0.2)

    train_native = train.to_native() if hasattr(train, "to_native") else train
    test_native = test.to_native() if hasattr(test, "to_native") else test

    last_train = train_native.groupby("user_id")["timestamp"].max()
    first_test = test_native.groupby("user_id")["timestamp"].min()
    shared = last_train.index.intersection(first_test.index)
    assert (last_train[shared] < first_test[shared]).all()


def test_a_seeded_split_is_reproducible(transactions: pd.DataFrame):
    """The same seed gives the same split, a different one does not."""
    first = split(transactions, strategy="random_holdout", ratio=0.2, seed=7)
    again = split(transactions, strategy="random_holdout", ratio=0.2, seed=7)
    other = split(transactions, strategy="random_holdout", ratio=0.2, seed=8)

    assert pairs(first[2]) == pairs(again[2])
    assert pairs(first[2]) != pairs(other[2])


def test_validation_split_comes_out_of_the_training_half(transactions: pd.DataFrame):
    """A validation split may only consume transactions the test split left."""
    train, validation, test = Splitter().split_transaction(
        transactions,
        labels=ColumnLabels(),
        test=SplitSpec(strategy="random_holdout", ratio=0.2, seed=3),
        validation=SplitSpec(strategy="random_holdout", ratio=0.1, seed=4),
    )

    assert pairs(validation) & pairs(test) == set()
    assert pairs(train) & pairs(validation) == set()
    assert pairs(train) & pairs(test) == set()


def test_cross_validation_returns_the_requested_folds(transactions: pd.DataFrame):
    """K-fold validation yields k disjoint validation sets."""
    _, folds, _ = Splitter().split_transaction(
        transactions,
        labels=ColumnLabels(),
        test=SplitSpec(strategy="random_holdout", ratio=0.2, seed=3),
        validation=SplitSpec(strategy="k_fold_cross_validation", folds=4, seed=4),
    )

    assert isinstance(folds, list) and len(folds) == 4
    validations = [pairs(validation) for _, validation in folds]
    for i, left in enumerate(validations):
        for right in validations[i + 1 :]:
            assert left & right == set(), "two folds share a transaction"
