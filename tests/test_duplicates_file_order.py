"""The 'first' and 'last' duplicate policies follow file order on every backend.

A repeated (user, item) pair collapses into one cell of the interaction matrix.
'first' and 'last' keep the rating of the earliest or latest row in the file,
but polars joins do not keep the input order, so on the default backend the
kept rating used to be an arbitrary one of the repeats.
"""

from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
import pytest

from warprec.data.dataset import Dataset
from warprec.utils.enums import RatingType

N_ROWS = 200_000
BACKENDS = [
    pytest.param(lambda frame: frame, id="pandas"),
    pytest.param(pl.from_pandas, id="polars"),
]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Many repeated pairs, each row rated with its position in the file.

    Returns:
        pd.DataFrame: The interactions.
    """
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, 1000, N_ROWS),
            "item_id": rng.integers(0, 300, N_ROWS),
            "rating": np.arange(N_ROWS, dtype=np.float64),
        }
    )


@pytest.mark.parametrize("policy", ["first", "last", "max", "mean", "sum"])
@pytest.mark.parametrize("to_backend", BACKENDS)
def test_duplicates_follow_file_order(
    frame: pd.DataFrame, to_backend: Callable[[pd.DataFrame], Any], policy: str
):
    """Every cell holds the rating the policy picks among the rows in file order.

    Args:
        frame (pd.DataFrame): The interactions.
        to_backend (Callable[[pd.DataFrame], Any]): Converts a frame to the backend.
        policy (str): The duplicate policy under test.
    """
    truth = frame.groupby(["user_id", "item_id"])["rating"].agg(policy)
    dataset = Dataset(
        train_data=to_backend(frame),
        rating_type=RatingType.EXPLICIT,
        rating_label="rating",
        duplicates=policy,
        batch_size=1024,
    )
    umap, imap = dataset.get_mappings()
    rows = [umap[user] for user in truth.index.get_level_values(0)]
    cols = [imap[item] for item in truth.index.get_level_values(1)]
    sparse = dataset.train_set.get_sparse()

    assert sparse.nnz == len(truth)
    got = np.asarray(sparse[rows, cols]).ravel()
    mismatches = int((~np.isclose(got, truth.to_numpy(), rtol=1e-12, atol=0)).sum())
    assert mismatches == 0, f"{mismatches} of {len(truth)} cells disagree"
