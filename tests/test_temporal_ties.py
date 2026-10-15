"""Rows sharing a timestamp keep their file order wherever time orders them.

Temporal order is (timestamp, then position in the file), and the most recent
interactions are the last rows in that order. ML-100K, like most logs, records
several interactions of a user in the same second, so a strategy that breaks
those ties some other way picks a different test item, keeps a different
history or feeds a sequential model a different sequence, and the two backends
used not to agree on which.
"""

from typing import Any, Callable, Dict, List, Set

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from warprec.utils.registry import filter_registry, splitting_registry

BACKENDS = [
    pytest.param(lambda frame: frame, id="pandas"),
    pytest.param(pl.from_pandas, id="polars"),
]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Shuffled interactions with few distinct timestamps, so ties abound.

    Every row carries its position in the file under ``pos``, which identifies it
    whatever a strategy does to the others.

    Returns:
        pd.DataFrame: The interactions.
    """
    rng = np.random.default_rng(0)
    n_users, per_user = 300, 12
    users = np.repeat(np.arange(n_users), per_user)
    items = np.concatenate(
        [rng.choice(100, per_user, replace=False) for _ in range(n_users)]
    )
    shuffle = rng.permutation(users.size)
    return pd.DataFrame(
        {
            "user_id": users[shuffle],
            "item_id": items[shuffle],
            "rating": 1.0,
            "timestamp": rng.integers(0, 3, users.size),
            "pos": np.arange(users.size),
        }
    )


def temporal_order(frame: pd.DataFrame) -> Dict[int, List[int]]:
    """The file positions of each user's rows, oldest first.

    Args:
        frame (pd.DataFrame): The interactions.

    Returns:
        Dict[int, List[int]]: The positions per user, in temporal order.
    """
    ordered = frame.sort_values(["user_id", "timestamp", "pos"], kind="stable")
    return {
        int(user): group["pos"].tolist()
        for user, group in ordered.groupby("user_id", sort=False)
    }


def positions(half: Any) -> Set[int]:
    """The file positions of the rows a frame holds.

    Args:
        half (Any): A narwhals frame.

    Returns:
        Set[int]: The positions.
    """
    return set(half["pos"].to_list())


def test_the_brief_example_holds_out_the_later_row_of_a_tie():
    """Items 12 and 13 share the latest second; 13 comes later in the file."""
    rows = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "item_id": [10, 11, 12, 13],
            "rating": 1.0,
            "timestamp": [100, 200, 300, 300],
        }
    )
    for native in (rows, pl.from_pandas(rows)):
        _, test = splitting_registry.get("temporal_leave_k_out")(
            nw.from_native(native), k=1
        )[0]
        assert test["item_id"].to_list() == [13]
        _, test = splitting_registry.get("temporal_holdout")(
            nw.from_native(native), ratio=0.25
        )[0]
        assert test["item_id"].to_list() == [13]


@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("to_backend", BACKENDS)
def test_temporal_leave_k_out_holds_out_the_last_k_rows(
    frame: pd.DataFrame, to_backend: Callable[[pd.DataFrame], Any], k: int
):
    """The test set holds exactly each user's last k rows in temporal order.

    Args:
        frame (pd.DataFrame): The interactions.
        to_backend (Callable[[pd.DataFrame], Any]): Converts a frame to the backend.
        k (int): The rows held out per user.
    """
    expected = {p for order in temporal_order(frame).values() for p in order[-k:]}

    train, test = splitting_registry.get("temporal_leave_k_out")(
        nw.from_native(to_backend(frame)), k=k
    )[0]

    assert positions(test) == expected
    assert positions(train) == set(frame["pos"]) - expected


@pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("to_backend", BACKENDS)
def test_temporal_holdout_holds_out_the_last_rows(
    frame: pd.DataFrame, to_backend: Callable[[pd.DataFrame], Any], ratio: float
):
    """The test set holds each user's most recent rows in temporal order.

    Args:
        frame (pd.DataFrame): The interactions.
        to_backend (Callable[[pd.DataFrame], Any]): Converts a frame to the backend.
        ratio (float): The fraction of each user's rows held out.
    """
    expected = set()
    for order in temporal_order(frame).values():
        keep = max(int(np.floor(len(order) * (1 - ratio))), 1)
        expected |= set(order[keep:])

    train, test = splitting_registry.get("temporal_holdout")(
        nw.from_native(to_backend(frame)), ratio=ratio
    )[0]

    assert positions(test) == expected
    assert positions(train) == set(frame["pos"]) - expected


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("name", ["UserHeadN", "UserTailN"])
@pytest.mark.parametrize("to_backend", BACKENDS)
def test_head_and_tail_filters_keep_the_first_or_last_rows(
    frame: pd.DataFrame, to_backend: Callable[[pd.DataFrame], Any], name: str, n: int
):
    """UserHeadN keeps each user's first n rows in temporal order, UserTailN the last.

    Args:
        frame (pd.DataFrame): The interactions.
        to_backend (Callable[[pd.DataFrame], Any]): Converts a frame to the backend.
        name (str): The registered filter name.
        n (int): The rows kept per user.
    """
    expected = {
        p
        for order in temporal_order(frame).values()
        for p in (order[:n] if name == "UserHeadN" else order[-n:])
    }

    kept = filter_registry.get(name, num_interactions=n)(
        nw.from_native(to_backend(frame))
    )

    assert positions(kept) == expected
