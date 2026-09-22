"""Behavioural tests for the cold-start protocols.

Every one of these guards a place where the pipeline would otherwise quietly
discard the entities the protocol exists to ask about. The failure mode is not an
exception but an empty evaluation set, so the assertions are about what survives.
"""

from typing import Any, Set

import numpy as np
import pandas as pd
import pytest

from warprec.data.dataset import Dataset
from warprec.data.ranking import cold_item_candidates
from warprec.data.schema import ColumnLabels, SplitSpec
from warprec.data.splitting import Splitter

N_USERS, N_ITEMS = 40, 60
PROTOCOLS = [
    ("item_cold_start", "item", "item_id"),
    ("user_cold_start", "user", "user_id"),
]


@pytest.fixture(scope="module")
def interactions() -> pd.DataFrame:
    """Interactions dense enough that holding entities out still leaves a train set.

    Returns:
        pd.DataFrame: The generated interactions.
    """
    rng = np.random.default_rng(11)
    rows = [
        (user, int(item))
        for user in range(N_USERS)
        for item in rng.choice(N_ITEMS, 12, replace=False)
    ]
    frame = pd.DataFrame(rows, columns=["user_id", "item_id"])
    frame["rating"] = 1.0
    frame["timestamp"] = np.arange(len(frame))
    return frame


@pytest.fixture(scope="module")
def attributes() -> pd.DataFrame:
    """Item attributes, so the side-information path is exercised too.

    Returns:
        pd.DataFrame: The generated attributes.
    """
    rng = np.random.default_rng(12)
    return pd.DataFrame(
        {
            "item_id": np.arange(N_ITEMS),
            **{f"f{i}": rng.integers(0, 2, N_ITEMS) for i in range(4)},
        }
    )


def values(frame: Any, column: str) -> Set:
    """The distinct values a column holds.

    Args:
        frame (Any): The frame to read, in any backend narwhals accepts.
        column (str): The column to read.

    Returns:
        Set: The distinct values.
    """
    native = frame.to_native() if hasattr(frame, "to_native") else frame
    return set(native[column].tolist())


@pytest.mark.parametrize("strategy,dimension,column", PROTOCOLS)
def test_the_held_out_entities_never_appear_in_training(
    interactions: pd.DataFrame, strategy: str, dimension: str, column: str
):
    """A cold entity keeps no history at all, which is the whole protocol."""
    spec = SplitSpec(strategy=strategy, ratio=0.25, seed=3)
    train, _, test = Splitter().split_transaction(
        interactions, labels=ColumnLabels(), test=spec
    )

    assert len(test.to_native() if hasattr(test, "to_native") else test) > 0
    assert values(train, column) & values(test, column) == set()


@pytest.mark.parametrize("strategy,dimension,column", PROTOCOLS)
def test_the_splitter_reports_which_side_is_held_out(
    strategy: str, dimension: str, column: str
):
    """The rest of the pipeline reads the protocol off the strategy."""
    assert Splitter().cold_dimension(SplitSpec(strategy=strategy)) == dimension


def test_an_ordinary_strategy_holds_nothing_out():
    """Only the cold-start strategies name a dimension."""
    assert Splitter().cold_dimension(SplitSpec(strategy="random_holdout")) is None


@pytest.mark.parametrize("ratio", [0.0, 1.0, 1.5])
def test_a_ratio_that_would_empty_a_side_is_refused(
    interactions: pd.DataFrame, ratio: float
):
    """A protocol that leaves nothing to train or test on is a configuration error."""
    with pytest.raises(ValueError):
        Splitter().split_transaction(
            interactions,
            labels=ColumnLabels(),
            test=SplitSpec(strategy="item_cold_start", ratio=ratio, seed=3),
        )


@pytest.mark.parametrize("strategy,dimension,column", PROTOCOLS)
def test_the_catalogue_keeps_the_entities_the_split_held_out(
    interactions: pd.DataFrame,
    attributes: pd.DataFrame,
    strategy: str,
    dimension: str,
    column: str,
):
    """A cold entity absent from the mappings could never be scored at all."""
    spec = SplitSpec(strategy=strategy, ratio=0.25, seed=3)
    cold = Splitter().cold_dimension(spec)
    train, _, test = Splitter().split_transaction(
        interactions, labels=ColumnLabels(), test=spec
    )

    dataset = Dataset(
        train_data=train,
        eval_data=test,
        side_data=attributes,
        rating_type="implicit",
        timestamp_label="timestamp",
        cold_start=cold,
        batch_size=16,
    )

    info = dataset.info()
    assert info["n_items"] == N_ITEMS
    assert info["n_users"] == N_USERS
    assert len(dataset.eval_set.get_df()) > 0


def test_without_the_protocol_the_evaluation_set_is_lost(
    interactions: pd.DataFrame, attributes: pd.DataFrame
):
    """The control: this is what the pipeline does when not told about the protocol.

    It is not an error, which is exactly why it needs a test. The cold items are
    filtered away and the evaluation set comes back empty.
    """
    spec = SplitSpec(strategy="item_cold_start", ratio=0.25, seed=3)
    train, _, test = Splitter().split_transaction(
        interactions, labels=ColumnLabels(), test=spec
    )

    dataset = Dataset(
        train_data=train,
        eval_data=test,
        side_data=attributes,
        rating_type="implicit",
        timestamp_label="timestamp",
        cold_start=None,
        batch_size=16,
    )

    assert dataset.info()["n_items"] < N_ITEMS
    assert len(dataset.eval_set.get_df()) == 0


def test_the_candidate_sets_split_the_catalogue(dataset: Dataset):
    """'cold' and 'warm' are complements, and 'all' restricts nothing."""
    train_sparse = dataset.train_set.get_sparse()

    cold = cold_item_candidates(train_sparse, "cold")
    warm = cold_item_candidates(train_sparse, "warm")

    assert cold_item_candidates(train_sparse, "all") is None
    assert cold is not None and warm is not None
    assert bool((cold ^ warm).all()), "an item was in both populations or neither"


def test_an_unknown_candidate_set_is_refused(dataset: Dataset):
    """A misspelled candidate set must not silently rank the whole catalogue."""
    with pytest.raises(ValueError, match="not supported"):
        cold_item_candidates(dataset.train_set.get_sparse(), "lukewarm")
