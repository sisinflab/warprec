"""Shared fixtures for the test suite.

Every fixture here is generated in memory. The datasets live outside the
repository, so a test that reads one would pass locally and fail in CI.
"""

from typing import List

import numpy as np
import pandas as pd
import pytest

from warprec.data.dataset import Dataset

N_USERS = 40
N_ITEMS = 25
N_INTERACTIONS = 600
CONTEXT_LABELS: List[str] = ["daytime", "weather"]


@pytest.fixture(scope="session")
def interactions_frame() -> pd.DataFrame:
    """A small, dense interaction frame with ratings, timestamps and contexts.

    Returns:
        pd.DataFrame: The generated interactions.
    """
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "user_id": rng.integers(0, N_USERS, N_INTERACTIONS),
            "item_id": rng.integers(0, N_ITEMS, N_INTERACTIONS),
            "rating": rng.integers(1, 6, N_INTERACTIONS).astype(float),
            "timestamp": rng.integers(1_000_000, 2_000_000, N_INTERACTIONS),
            "daytime": rng.choice(["morning", "evening"], N_INTERACTIONS),
            "weather": rng.choice(["sunny", "rainy"], N_INTERACTIONS),
        }
    ).drop_duplicates(subset=["user_id", "item_id"])

    # Every user needs at least two interactions so that a split leaves a
    # training history behind, and every item needs to be reachable.
    filler = pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(N_USERS), 2),
            "item_id": np.tile(np.arange(N_ITEMS), 4)[: N_USERS * 2],
            "rating": 4.0,
            "timestamp": 1_500_000,
            "daytime": "morning",
            "weather": "sunny",
        }
    )
    frame = pd.concat([frame, filler]).drop_duplicates(subset=["user_id", "item_id"])
    return frame.reset_index(drop=True)


@pytest.fixture(scope="session")
def side_frame() -> pd.DataFrame:
    """Item attributes in the wide, one-hot layout the content models expect.

    Returns:
        pd.DataFrame: The generated item features.
    """
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "item_id": np.arange(N_ITEMS),
            "action": rng.integers(0, 2, N_ITEMS),
            "comedy": rng.integers(0, 2, N_ITEMS),
            "drama": rng.integers(0, 2, N_ITEMS),
        }
    )


@pytest.fixture(scope="session")
def dataset(interactions_frame: pd.DataFrame, side_frame: pd.DataFrame) -> Dataset:
    """A Dataset carrying everything the model families need at once.

    Contexts and side information are both present so that a single fixture
    serves the collaborative, content, context-aware, sequential and hybrid
    families without special-casing.

    Args:
        interactions_frame (pd.DataFrame): The generated interactions.
        side_frame (pd.DataFrame): The generated item features.

    Returns:
        Dataset: The dataset under test.
    """
    train = interactions_frame.groupby("user_id", group_keys=False).apply(
        lambda g: g.iloc[:-1]
    )
    evaluation = interactions_frame.groupby("user_id", group_keys=False).apply(
        lambda g: g.iloc[-1:]
    )
    return Dataset(
        train_data=train,
        eval_data=evaluation,
        side_data=side_frame,
        rating_type="explicit",
        rating_label="rating",
        timestamp_label="timestamp",
        context_labels=CONTEXT_LABELS,
        batch_size=64,
    )
