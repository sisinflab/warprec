"""Behavioural tests for reading and writing transaction data.

The reader is the only place a user's file meets WarpRec's schema, so the
combinations that change how a file is parsed are exercised directly: with and
without a header, both separators, both file formats and both backends. The
writer is checked by reading back what it wrote.
"""

from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import pytest
import torch

from warprec.data.dataset import Dataset
from warprec.data.reader import LocalReader
from warprec.data.writer import LocalWriter
from warprec.utils.registry import model_registry

COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
BACKENDS = ["pandas", "polars"]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """A small transaction frame with every core column populated.

    Returns:
        pd.DataFrame: The generated transactions.
    """
    rng = np.random.default_rng(5)
    n = 120
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, 15, n),
            "item_id": rng.integers(0, 25, n),
            "rating": rng.integers(1, 6, n).astype(float),
            "timestamp": np.arange(n) * 7,
        }
    )


def as_pandas(read: Any) -> pd.DataFrame:
    """Bring whatever the reader returned back to pandas for comparison.

    Args:
        read (Any): The frame the reader produced.

    Returns:
        pd.DataFrame: The same rows, as pandas.
    """
    native = read.to_native() if hasattr(read, "to_native") else read
    if not isinstance(native, pd.DataFrame):
        native = native.to_pandas()
    return native.reset_index(drop=True)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sep", [",", "\t", ";"])
@pytest.mark.parametrize("header", [True, False])
def test_tabular_reading_round_trips(
    tmp_path: Path, frame: pd.DataFrame, backend: str, sep: str, header: bool
):
    """A file written with any separator reads back with the same values."""
    path = tmp_path / f"data_{backend}_{header}.csv"
    frame.to_csv(path, sep=sep, header=header, index=False)

    read = LocalReader(backend=backend).read_tabular(
        local_path=str(path),
        sep=sep,
        header=header,
        # Without a header the reader is told the schema, which is the
        # documented contract for a headerless file.
        column_names=None if header else COLUMNS,
    )

    out = as_pandas(read)
    assert list(out.columns) == COLUMNS
    assert len(out) == len(frame)
    np.testing.assert_array_equal(
        out["user_id"].to_numpy().astype(np.int64),
        frame["user_id"].to_numpy().astype(np.int64),
    )
    np.testing.assert_allclose(
        out["rating"].to_numpy().astype(float), frame["rating"].to_numpy()
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_parquet_reading_round_trips(tmp_path: Path, frame: pd.DataFrame, backend: str):
    """Parquet carries the same rows as the tabular path."""
    path = tmp_path / "data.parquet"
    frame.to_parquet(path, index=False)

    read = LocalReader(backend=backend).read_parquet(local_path=str(path))

    out = as_pandas(read)
    assert list(out.columns) == COLUMNS
    assert len(out) == len(frame)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_missing_file_reads_as_empty_rather_than_raising(
    tmp_path: Path, backend: str
):
    """A path that does not exist yields an empty frame, not an exception."""
    read = LocalReader(backend=backend).read_tabular(
        local_path=str(tmp_path / "absent.csv")
    )

    assert len(as_pandas(read)) == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_custom_labels_are_mapped_onto_the_schema(
    tmp_path: Path, frame: pd.DataFrame, backend: str
):
    """A file with its own column names is read through column_names."""
    renamed = frame.rename(
        columns={"user_id": "uid", "item_id": "iid", "timestamp": "time_ms"}
    )
    path = tmp_path / "renamed.csv"
    renamed.to_csv(path, sep=",", header=False, index=False)

    read = LocalReader(backend=backend).read_tabular(
        local_path=str(path),
        sep=",",
        header=False,
        column_names=["uid", "iid", "rating", "time_ms"],
    )

    out = as_pandas(read)
    assert list(out.columns) == ["uid", "iid", "rating", "time_ms"]
    assert len(out) == len(frame)


@pytest.mark.parametrize("ext,sep", [(".tsv", "\t"), (".csv", ",")])
def test_a_written_split_reads_back(
    tmp_path: Path, frame: pd.DataFrame, ext: str, sep: str
):
    """What the writer wrote is what the reader gets back."""
    train = frame.iloc[:90]
    test = frame.iloc[90:]
    dataset = Dataset(
        train_data=train,
        eval_data=test,
        rating_type="explicit",
        rating_label="rating",
        timestamp_label="timestamp",
    )

    writer = LocalWriter(dataset_name="roundtrip", local_path=str(tmp_path))
    writer.write_tabular_split(dataset, None, None, sep=sep, ext=ext, header=True)

    split_dir = Path(writer.experiment_split_path)
    written: List[str] = sorted(p.name for p in split_dir.rglob(f"*{ext}"))
    assert "train" + ext in written and "test" + ext in written

    reader = LocalReader(backend="pandas")
    back = as_pandas(
        reader.read_tabular(
            local_path=str(next(split_dir.rglob("train" + ext))), sep=sep, header=True
        )
    )
    assert len(back) == len(train)
    assert set(back.columns) >= {"user_id", "item_id"}


def recommended_pairs(writer: LocalWriter, sep: str = "\t") -> pd.DataFrame:
    """The recommendations a writer produced.

    Args:
        writer (LocalWriter): The writer that wrote them.
        sep (str): The separator the file was written with.

    Returns:
        pd.DataFrame: The recommendation rows.
    """
    written = sorted(Path(writer.experiment_recommendation_path).glob("*.tsv"))
    return pd.read_csv(written[0], sep=sep)


def seen_among(recommendations: pd.DataFrame, dataset: Dataset) -> int:
    """How many recommended items the user had already interacted with.

    Args:
        recommendations (pd.DataFrame): The written recommendations.
        dataset (Dataset): The dataset the model was trained on.

    Returns:
        int: The number of already-seen recommendations.
    """
    train = dataset.train_set.get_sparse()
    users = dataset.info()["user_mapping"]
    items = dataset.info()["item_mapping"]

    total = 0
    for user, item in zip(recommendations["user_id"], recommendations["item_id"]):
        row, column = users.get(user), items.get(item)
        if row is not None and column is not None and train[row, column] > 0:
            total += 1
    return total


def write_with(tmp_path: Path, dataset: Dataset, mask_seen: str) -> pd.DataFrame:
    """Write recommendations under one seen-item rule.

    Args:
        tmp_path (Path): The directory pytest provides.
        dataset (Dataset): The dataset under test.
        mask_seen (str): The rule to apply.

    Returns:
        pd.DataFrame: The written recommendations.
    """
    torch.manual_seed(42)
    model = model_registry.get(
        "ITEMKNN",
        params={"k": 20, "similarity": "cosine"},
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
        seed=42,
    )
    writer = LocalWriter(dataset_name=f"recs_{mask_seen}", local_path=str(tmp_path))
    writer.write_recs(model=model, dataset=dataset, k=5, mask_seen=mask_seen)
    return recommended_pairs(writer)


def test_written_recommendations_exclude_seen_items_by_default(
    tmp_path: Path, dataset: Dataset
):
    """The pair rule keeps everything the user already interacted with out."""
    assert seen_among(write_with(tmp_path, dataset, "pair"), dataset) == 0


def test_written_recommendations_honour_a_run_that_masks_nothing(
    tmp_path: Path, dataset: Dataset
):
    """'none' has to reach the written list, not only the evaluation.

    Before this was wired the recommendations were filtered whatever the run
    asked for, so the file on disk was ranked under a different rule from the
    numbers reported beside it.
    """
    assert seen_among(write_with(tmp_path, dataset, "none"), dataset) > 0


def test_a_contextual_rule_falls_back_rather_than_being_ignored(
    tmp_path: Path, dataset: Dataset
):
    """There is no situation to compare against when writing, so 'pair' applies.

    The fixture carries contextual columns, so 'auto' resolves to the contextual
    rule for evaluation and has to fall back here.
    """
    assert dataset.train_transactions is not None
    assert seen_among(write_with(tmp_path, dataset, "auto"), dataset) == 0
