"""A fixed seed has to produce a fixed run.

Every sampling dataloader is built twice with the same seed and once with a
different one. Two runs at the same seed must agree batch for batch, and two
runs at different seeds must not, which is what stops the seed being honoured
in name only. The defect this guards against was real: the samplers drew from
numpy's global generator, which nothing seeds when the loader runs in the main
process, so two identical runs disagreed.
"""

from typing import Any, Callable, List

import numpy as np
import pandas as pd
import pytest
import torch

from warprec.data.dataset import Dataset

N_USERS, N_ITEMS = 30, 45


@pytest.fixture(scope="module")
def sampling_dataset() -> Dataset:
    """A dataset dense enough for every sequential and pointwise loader.

    Returns:
        Dataset: The dataset under test.
    """
    rng = np.random.default_rng(19)
    rows = []
    for user in range(N_USERS):
        for item in rng.choice(N_ITEMS, 12, replace=False):
            rows.append((user, int(item)))
    frame = pd.DataFrame(rows, columns=["user_id", "item_id"])
    frame["rating"] = 1.0
    frame["timestamp"] = np.arange(len(frame))
    frame["mood"] = rng.choice(["a", "b"], len(frame))

    train = frame.groupby("user_id", group_keys=False).apply(lambda g: g.iloc[:-1])
    evaluation = frame.groupby("user_id", group_keys=False).apply(lambda g: g.iloc[-1:])
    return Dataset(
        train_data=train,
        eval_data=evaluation,
        rating_type="implicit",
        timestamp_label="timestamp",
        context_labels=["mood"],
        batch_size=16,
    )


def batches(loader: Any, limit: int = 5) -> List[np.ndarray]:
    """The first few batches of a loader, flattened for comparison.

    Args:
        loader (Any): The dataloader to drain.
        limit (int): How many batches to take.

    Returns:
        List[np.ndarray]: One array per tensor of each batch.
    """
    out: List[np.ndarray] = []
    for i, batch in enumerate(loader):
        parts = batch if isinstance(batch, (list, tuple)) else [batch]
        out.extend(np.asarray(t) for t in parts)
        if i + 1 >= limit:
            break
    return out


def assert_same(left: List[np.ndarray], right: List[np.ndarray]) -> None:
    """Assert two drained loaders agree everywhere.

    Args:
        left (List[np.ndarray]): The first run.
        right (List[np.ndarray]): The second run.
    """
    assert len(left) == len(right)
    for a, b in zip(left, right):
        np.testing.assert_array_equal(a, b)


def differs(left: List[np.ndarray], right: List[np.ndarray]) -> bool:
    """Whether two drained loaders disagree anywhere.

    Args:
        left (List[np.ndarray]): The first run.
        right (List[np.ndarray]): The second run.

    Returns:
        bool: True when any batch differs.
    """
    if len(left) != len(right):
        return True
    return any(
        a.shape != b.shape or not np.array_equal(a, b) for a, b in zip(left, right)
    )


def loaders(dataset: Dataset) -> List[tuple]:
    """Every loader that draws negatives or masks, with its builder.

    Args:
        dataset (Dataset): The dataset under test.

    Returns:
        List[tuple]: Pairs of name and a builder taking a seed.
    """
    inter, session, transactions = (
        dataset.train_set,
        dataset.train_session,
        dataset.train_transactions,
    )
    return [
        ("pointwise", lambda s: inter.get_pointwise_dataloader(neg_samples=2, seed=s)),
        ("contrastive", lambda s: inter.get_contrastive_dataloader(seed=s)),
        (
            "context pointwise",
            lambda s: transactions.get_pointwise_dataloader(
                neg_samples=2, include_context=True, seed=s
            ),
        ),
        (
            "sequential",
            lambda s: session.get_sequential_dataloader(
                max_seq_len=6, neg_samples=1, seed=s
            ),
        ),
        (
            "same target",
            lambda s: session.get_same_target_sequential_dataloader(
                max_seq_len=6, seed=s
            ),
        ),
        (
            "sliding window",
            lambda s: session.get_sliding_window_dataloader(
                max_seq_len=6, neg_samples=1, seed=s
            ),
        ),
        (
            "cloze",
            lambda s: session.get_cloze_mask_dataloader(
                max_seq_len=6,
                mask_prob=0.5,
                mask_token_id=N_ITEMS,
                neg_samples=1,
                seed=s,
            ),
        ),
    ]


@pytest.mark.parametrize(
    "name",
    [
        "pointwise",
        "contrastive",
        "context pointwise",
        "sequential",
        "same target",
        "sliding window",
        "cloze",
    ],
)
def test_the_same_seed_gives_the_same_batches(sampling_dataset: Dataset, name: str):
    """Two loaders built with one seed must agree batch for batch."""
    build: Callable = dict(loaders(sampling_dataset))[name]

    torch.manual_seed(0)
    first = batches(build(1234))
    torch.manual_seed(0)
    second = batches(build(1234))

    assert_same(first, second)


@pytest.mark.parametrize(
    "name",
    [
        "pointwise",
        "contrastive",
        "context pointwise",
        "sequential",
        "sliding window",
        "cloze",
    ],
)
def test_a_different_seed_gives_different_batches(sampling_dataset: Dataset, name: str):
    """A seed that is honoured in name only would pass the test above too."""
    build: Callable = dict(loaders(sampling_dataset))[name]

    torch.manual_seed(0)
    first = batches(build(1234))
    torch.manual_seed(0)
    other = batches(build(4321))

    assert differs(first, other), f"{name} ignored its seed"


def test_sampling_does_not_touch_the_global_generator(sampling_dataset: Dataset):
    """Draining a loader must leave numpy's global stream where it found it."""
    np.random.seed(7)
    before = np.random.random()

    np.random.seed(7)
    batches(sampling_dataset.train_set.get_pointwise_dataloader(neg_samples=2, seed=1))
    after = np.random.random()

    assert before == after
