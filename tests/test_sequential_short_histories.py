"""Every sequential model must score a batch whatever its longest history is.

The evaluator pads each batch only as wide as the longest history in it, so a
batch is narrower than max_seq_len whenever none of its users has that many
interactions, and one column wide when none has any, which is every batch of a
user cold-start protocol. A model that assumes the full window, through a mask
or a filter built for max_seq_len, crashes on those batches, and a model whose
output depends on how much padding follows a history ranks the same user
differently depending on who else is in the batch.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import (
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.utils.registry import model_registry

from conftest import build_params

SEQUENTIAL = sorted(
    name
    for name in model_registry.list_registered()
    if issubclass(model_registry.get_class(name), SequentialRecommenderUtils)
)

# Every model in its tested configuration, plus the switches that send a
# model's forward pass through a different encoder.
CASES = [pytest.param(name, {}, id=name) for name in SEQUENTIAL] + [
    pytest.param("CORE", {"dnn_type": "ave"}, id="CORE-ave"),
    pytest.param("ESASREC", {"use_ligr": False}, id="ESASREC-no-ligr"),
    pytest.param("GSASREC", {"reuse_item_embeddings": False}, id="GSASREC-own-out"),
]
MAX_SEQ_LEN = 6
N_ITEMS = 30

# Training history length of each user. The cold ones appear only in the
# evaluation data, the way a user cold-start split leaves them.
HISTORY: Dict[int, int] = {
    0: 12,
    1: 9,
    2: 7,
    3: 4,
    4: 3,
    5: 2,
    6: 1,
    7: 0,
    8: 0,
    9: 0,
}
LONG = [0, 1, 2]
SHORT = [3, 4, 5, 6]
COLD = [7, 8, 9]


def test_every_sequential_model_is_covered():
    """Guards against the discovery above silently matching fewer models."""
    assert len(SEQUENTIAL) == 15, SEQUENTIAL


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """Users with long, short and no training histories.

    Returns:
        Dataset: The dataset under test, built as for a user cold-start split.
    """
    rng = np.random.default_rng(5)
    train_rows, eval_rows = [], []
    clock = 0
    for user, length in HISTORY.items():
        items = rng.choice(N_ITEMS, length + 2, replace=False)
        for item in items[:length]:
            train_rows.append((user, int(item), 1.0, clock))
            clock += 1
        for item in items[length:]:
            eval_rows.append((user, int(item), 1.0, clock))
            clock += 1

    # Every item has to be in training so that the catalogue is the same one
    # the models are built over.
    for item in range(N_ITEMS):
        train_rows.append((10 + item % 3, item, 1.0, clock))
        clock += 1

    columns = ["user_id", "item_id", "rating", "timestamp"]
    return Dataset(
        train_data=pd.DataFrame(train_rows, columns=columns),
        eval_data=pd.DataFrame(eval_rows, columns=columns),
        rating_type="implicit",
        timestamp_label="timestamp",
        cold_start="user",
        batch_size=16,
    )


def build(model_name: str, overrides: Dict[str, Any], dataset: Dataset) -> Recommender:
    """Build an untrained model with a window wider than the short histories.

    Args:
        model_name (str): The registered name.
        overrides (Dict[str, Any]): Hyperparameters set on top of the defaults.
        dataset (Dataset): The dataset under test.

    Returns:
        Recommender: The model, in evaluation mode.
    """
    params = build_params(model_name)
    params.update(overrides, max_seq_len=MAX_SEQ_LEN)
    torch.manual_seed(0)
    model = model_registry.get(
        model_name,
        params=params,
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
    )
    model.eval()
    return model


def score(
    model: Recommender, dataset: Dataset, users: List[int]
) -> Tuple[torch.Tensor, int]:
    """Score users exactly as the evaluator does.

    Args:
        model (Recommender): The model to ask.
        dataset (Dataset): The dataset under test.
        users (List[int]): The original user identifiers to score together.

    Returns:
        Tuple[torch.Tensor, int]: The scores, and the width of the padded batch.
    """
    mapping = dataset.info()["user_mapping"]
    indices = [mapping[user] for user in users]
    user_seq, seq_len = dataset.train_session.get_user_history_sequences(
        indices, model.max_seq_len
    )
    with torch.inference_mode():
        scores = model.predict(
            user_indices=torch.tensor(indices),
            user_seq=user_seq,
            seq_len=seq_len,
        )
    return scores, user_seq.shape[1]


@pytest.mark.parametrize("model_name,overrides", CASES)
def test_a_batch_of_users_without_history_is_scored(
    model_name: str, overrides: Dict[str, Any], dataset: Dataset
):
    """A user cold-start batch is one column of padding and must still rank."""
    model = build(model_name, overrides, dataset)
    scores, width = score(model, dataset, COLD)

    assert width == 1
    assert scores.shape == (len(COLD), dataset.info()["n_items"])
    assert torch.isfinite(scores).all(), f"{model_name}: non-finite cold scores"


@pytest.mark.parametrize("model_name,overrides", CASES)
def test_a_batch_shorter_than_the_window_is_scored(
    model_name: str, overrides: Dict[str, Any], dataset: Dataset
):
    """A batch whose longest history is under max_seq_len must still rank."""
    model = build(model_name, overrides, dataset)
    scores, width = score(model, dataset, SHORT)

    assert 1 < width < MAX_SEQ_LEN
    assert scores.shape == (len(SHORT), dataset.info()["n_items"])
    assert torch.isfinite(scores).all(), f"{model_name}: non-finite short scores"


@pytest.mark.parametrize("model_name,overrides", CASES)
def test_a_user_ranks_the_same_whoever_shares_the_batch(
    model_name: str, overrides: Dict[str, Any], dataset: Dataset
):
    """The width a batch is padded to must not change any user's scores.

    Each short and cold user is scored once among users as short as they are,
    and once next to users whose histories fill the window. The two have to
    agree, or the reported metrics would depend on how users were batched.
    """
    model = build(model_name, overrides, dataset)
    full, _ = score(model, dataset, LONG)
    again, _ = score(model, dataset, LONG)
    if not torch.equal(full, again):
        pytest.skip(f"{model_name} does not score reproducibly")

    narrow_short, _ = score(model, dataset, SHORT)
    narrow_cold, _ = score(model, dataset, COLD)
    mixed, width = score(model, dataset, LONG + SHORT + COLD)

    assert width == MAX_SEQ_LEN
    assert torch.isfinite(mixed).all(), f"{model_name}: non-finite mixed scores"
    torch.testing.assert_close(
        mixed[: len(LONG)], full, msg=f"{model_name}: long users moved"
    )
    torch.testing.assert_close(
        mixed[len(LONG) : len(LONG) + len(SHORT)],
        narrow_short,
        msg=f"{model_name}: short users depend on the batch width",
    )
    torch.testing.assert_close(
        mixed[len(LONG) + len(SHORT) :],
        narrow_cold,
        msg=f"{model_name}: cold users depend on the batch width",
    )
