"""The sliding-window loader (gSASRec's) cuts every window from the user's own rows.

A user's session keeps every training row, repeats of an item included, but
the window end used to come from the deduplicated interaction matrix, so a
repeat anywhere shortened the windows of that user and of every later one.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch

from warprec.data.dataset import Dataset

MAX_SEQ_LEN = 5


def _windows(dataset: Dataset, max_seq_len: int, stride: int) -> torch.Tensor:
    """Every window the loader yields, in order.

    Args:
        dataset (Dataset): The dataset whose training sessions are cut.
        max_seq_len (int): The window width.
        stride (int): The step between two windows of a user.

    Returns:
        torch.Tensor: The windows, [windows, max_seq_len].
    """
    loader = dataset.train_session.get_sliding_window_dataloader(
        max_seq_len=max_seq_len, neg_samples=0, stride=stride, shuffle=False
    )
    return torch.cat([batch[0] for batch in loader])


def _expected(
    histories: Dict[int, List[int]], max_seq_len: int, stride: int, padding: int
) -> torch.Tensor:
    """The windows cut straight from each user's history.

    Args:
        histories (Dict[int, List[int]]): Item indices per user index, in order.
        max_seq_len (int): The window width.
        stride (int): The step between two windows of a user.
        padding (int): The padding item index.

    Returns:
        torch.Tensor: The windows, [windows, max_seq_len].
    """
    rows = []
    for user in sorted(histories):
        history = histories[user]
        if len(history) < 2:
            continue
        for window in range(max(len(history) - max_seq_len, 0) // stride + 1):
            cut = history[window * stride : window * stride + max_seq_len]
            rows.append(cut + [padding] * (max_seq_len - len(cut)))
    return torch.tensor(rows, dtype=torch.long)


def test_the_reported_case_keeps_every_last_item():
    """The brief's repro: both users used to lose their last item."""
    dataset = Dataset(
        train_data=pl.DataFrame(
            {
                "user_id": [1, 1, 1, 1, 2, 2, 2],
                "item_id": [10, 20, 20, 30, 40, 50, 60],
                "timestamp": [1, 2, 3, 4, 1, 2, 3],
            }
        ),
        timestamp_label="timestamp",
    )
    windows = _windows(dataset, MAX_SEQ_LEN, stride=1)
    assert windows.tolist() == [[0, 1, 1, 2, 6], [3, 4, 5, 6, 6]]


@pytest.mark.parametrize("repeats", [True, False], ids=["repeats", "no-repeats"])
@pytest.mark.parametrize("max_seq_len,stride", [(5, 1), (3, 1), (3, 2)])
def test_windows_are_cut_from_each_users_history(
    repeats: bool, max_seq_len: int, stride: int
):
    """Every window matches the user's history, whether items repeat or not.

    Args:
        repeats (bool): Whether users interact with an item more than once.
        max_seq_len (int): The window width.
        stride (int): The step between two windows of a user.
    """
    rng = np.random.default_rng(1)
    rows = []
    for user in range(12):
        length = int(rng.integers(1, 10))
        items = (
            rng.integers(0, 8, length)
            if repeats
            else rng.choice(20, length, replace=False)
        )
        rows += [(user, int(item), step) for step, item in enumerate(items)]
    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])
    dataset = Dataset(train_data=frame, timestamp_label="timestamp")

    umap, imap = dataset.get_mappings()
    histories: Dict[int, List[int]] = {}
    for user, item, _ in rows:
        histories.setdefault(umap[user], []).append(imap[item])
    expected = _expected(histories, max_seq_len, stride, dataset.info()["n_items"])

    assert torch.equal(_windows(dataset, max_seq_len, stride), expected)
