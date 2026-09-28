"""What each entity can and cannot put in a pointwise training dataset.

Both entities build one, and on plain data they build the same one, field for
field. They are not interchangeable though, and the two ways they differ are of
very different kinds. Repeated pairs are a real difference: a matrix has one
cell per pair and aggregates them, a transaction is a record and is kept.
Contexts were not a difference at all but a defect, and the matrix can no longer
be asked for them.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from warprec.data.dataset import Dataset

# Deliberately not in sorted order, and with (1, 3) recorded twice.
ROWS = [
    (2, 7),
    (0, 5),
    (2, 1),
    (1, 9),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 8),
    (1, 6),
    (1, 3),
]


def build(rows: list) -> Dataset:
    """A contextual dataset over the given training rows.

    Args:
        rows (list): The (user, item) pairs to train on.

    Returns:
        Dataset: The dataset under test.
    """
    frame = pd.DataFrame(rows, columns=["user_id", "item_id"])
    frame["timestamp"] = np.arange(len(frame))
    # A situation per row, so a misalignment shows up as a wrong value.
    frame["daypart"] = np.arange(len(frame)) % 3

    evaluation = pd.DataFrame([(0, 5), (1, 9), (2, 7)], columns=["user_id", "item_id"])
    evaluation["timestamp"] = np.arange(len(evaluation))
    evaluation["daypart"] = 0

    return Dataset(
        train_data=frame,
        eval_data=evaluation,
        rating_type="implicit",
        timestamp_label="timestamp",
        context_labels=["daypart"],
        batch_size=4,
    )


@pytest.fixture(name="dataset", scope="module")
def dataset_fixture() -> Dataset:
    """The shared fixture.

    Returns:
        Dataset: The dataset under test.
    """
    return build(ROWS)


def pairs(users: Any, items: Any) -> list:
    """Zip two index arrays into comparable pairs.

    Args:
        users (Any): The user of each row, as a tensor or an array.
        items (Any): The item of each row, as a tensor or an array.

    Returns:
        list: The (user, item) pairs in order.
    """
    return list(zip(np.asarray(users).tolist(), np.asarray(items).tolist()))


def test_both_entities_order_their_rows_the_same_way(dataset: Dataset):
    """Interactions rebuilds from a sorted matrix and Transactions sorts on the way in.

    Neither order is the order of the file. They agree, which is why the two
    entities produce the same dataset wherever they produce one at all.
    """
    from_interactions = pairs(*dataset.train_set._get_mapped_indices())
    transactions = dataset.train_transactions
    from_transactions = pairs(transactions._users, transactions._items)

    assert from_interactions == sorted(from_interactions)
    assert from_transactions == sorted(from_transactions)
    assert set(from_interactions) == set(from_transactions)


def test_interactions_collapses_a_repeated_pair_and_transactions_keeps_it(
    dataset: Dataset,
):
    """The real difference between the two, and the reason both exist.

    An interaction matrix has one cell per pair, so a pair recorded twice is
    aggregated by the `duplicates` policy. A transaction is a record, and that
    entity exists to preserve it.
    """
    transactions = dataset.train_transactions
    from_interactions = pairs(*dataset.train_set._get_mapped_indices())
    from_transactions = pairs(transactions._users, transactions._items)

    assert len(from_transactions) == len(ROWS), "a record was dropped"
    assert len(from_interactions) == len(set(ROWS)), "a duplicate cell survived"
    assert len(from_transactions) > len(from_interactions)


def test_without_repeats_the_two_datasets_are_identical():
    """With no pair recorded twice, the two build the same thing field for field."""
    dataset = build(list(dict.fromkeys(ROWS)))
    options = dict(neg_samples=1, batch_size=4, shuffle=False, seed=3)

    from_interactions = dataset.train_set.get_pointwise_dataloader(**options).dataset
    from_transactions = dataset.train_transactions.get_pointwise_dataloader(
        **options
    ).dataset

    assert pairs(from_interactions.user_ids, from_interactions.item_ids) == pairs(
        from_transactions.user_ids, from_transactions.item_ids
    )
    assert np.array_equal(
        from_interactions.sparse_matrix.indptr, from_transactions.sparse_matrix.indptr
    )
    assert np.array_equal(
        from_interactions.sparse_matrix.indices, from_transactions.sparse_matrix.indices
    )
    assert from_interactions.total_samples == from_transactions.total_samples


def test_the_matrix_refuses_to_supply_contexts(dataset: Dataset):
    """It used to supply them, attached to the wrong interactions.

    Interactions takes its rows from the matrix, which is sorted by user and
    then item, but it built the context array from the frame in the order the
    file happened to be in. The two lined up only by coincidence: on MovieLens
    100K with four distinct values, 19,816 of 80,000 rows agreed with the
    contexts Transactions gives, which is chance. A model trained that way
    learned situations belonging to other people's interactions.
    """
    with pytest.raises(NotImplementedError, match="cannot supply contexts"):
        dataset.train_set.get_pointwise_dataloader(neg_samples=1, include_context=True)


def test_the_records_entity_still_supplies_them(dataset: Dataset):
    """The refusal must point somewhere that works."""
    loader = dataset.train_transactions.get_pointwise_dataloader(
        neg_samples=0, batch_size=4, shuffle=False, seed=3, include_context=True
    )

    assert loader.dataset.contexts is not None
    assert len(loader.dataset.contexts) == len(ROWS)
