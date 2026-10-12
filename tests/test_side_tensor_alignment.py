"""The item-feature lookup of a split must be aligned with the catalogue.

Context-aware models register ``train_set.get_side_tensor()`` as their item
features and index it by item index. When the catalogue holds items a split
does not contain (``keep_unseen_items``, held-out cold items, or simply the
evaluation split), a lookup built from the split's own items is shorter than
the catalogue and every row after the first missing item belongs to another
item.
"""

from typing import Any, Callable

import pandas as pd
import polars as pl
import pytest
import torch

from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.registry import model_registry

from conftest import build_params

# Items 1, 4, 6 and 11 carry attributes but no interaction; item 3 appears only
# in the evaluation split.
TRAIN = pd.DataFrame(
    {
        "user_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "item_id": [2, 5, 7, 2, 8, 9, 5, 7, 10, 8, 9, 10],
        "timestamp": list(range(12)),
    }
)
EVAL = pd.DataFrame(
    {"user_id": [0, 1, 2, 3], "item_id": [3, 5, 2, 7], "timestamp": [20] * 4}
)
SIDE = pd.DataFrame(
    {
        "item_id": list(range(1, 12)),
        "genre": [f"g{i}" for i in range(1, 12)],
        "decade": [f"d{i % 4}" for i in range(1, 12)],
    }
)
BACKENDS = [
    pytest.param(lambda frame: frame, id="pandas"),
    pytest.param(pl.from_pandas, id="polars"),
]


def _build(to_backend: Callable[[pd.DataFrame], Any], keep: bool) -> Dataset:
    """Build the dataset on one backend.

    Args:
        to_backend (Callable[[pd.DataFrame], Any]): Converts a frame to the backend.
        keep (bool): Whether items without interactions stay in the catalogue.

    Returns:
        Dataset: The dataset under test.
    """
    return Dataset(
        train_data=to_backend(TRAIN),
        eval_data=to_backend(EVAL),
        side_data=to_backend(SIDE),
        timestamp_label="timestamp",
        keep_unseen_items=keep,
        batch_size=4,
    )


def _expected(dataset: Dataset) -> torch.Tensor:
    """The lookup each split must hold, built straight from the side table.

    Args:
        dataset (Dataset): The dataset whose catalogue the rows follow.

    Returns:
        torch.Tensor: One row per item in index order, then a zero padding row.
    """
    _, inverse = dataset.get_inverse_mappings()
    feature_maps = dataset._feature_maps  # pylint: disable=protected-access
    by_item = SIDE.set_index("item_id")
    rows = [
        [
            feature_maps[col][by_item.loc[inverse[idx], col]]
            for col in ("genre", "decade")
        ]
        for idx in range(len(inverse))
    ]
    return torch.tensor(rows + [[0, 0]], dtype=torch.long)


@pytest.mark.parametrize("keep", [True, False], ids=["keep", "drop"])
@pytest.mark.parametrize("to_backend", BACKENDS)
def test_split_lookups_follow_the_catalogue(
    to_backend: Callable[[pd.DataFrame], Any], keep: bool
):
    """Every split's lookup has a row per catalogue item, in item index order.

    Args:
        to_backend (Callable[[pd.DataFrame], Any]): Converts a frame to the backend.
        keep (bool): Whether items without interactions stay in the catalogue.
    """
    dataset = _build(to_backend, keep)
    n_items = dataset.info()["n_items"]
    assert n_items == (11 if keep else 6)

    expected = _expected(dataset)
    lookup = dataset.get_features_lookup().long()
    torch.testing.assert_close(lookup, expected, rtol=0, atol=0)
    for split in (dataset.train_set, dataset.eval_set):
        side = split.get_side_tensor()
        assert side.shape == (n_items + 1, 2)
        torch.testing.assert_close(side, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "model_name",
    ["AFM", "DCN", "DCNv2", "DeepFM", "FM", "NFM", "WideAndDeep", "xDeepFM"],
)
def test_context_models_train_on_a_widened_catalogue(model_name: str):
    """The context-aware models embed item features for every catalogue item.

    Args:
        model_name (str): The context-aware model under test.
    """
    dataset = _build(lambda frame: frame, keep=True)
    torch.manual_seed(0)
    model = model_registry.get(
        model_name,
        params=build_params(model_name),
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
    )
    assert isinstance(model, IterativeRecommender)
    torch.testing.assert_close(model.item_features, _expected(dataset), rtol=0, atol=0)
    loader = model.get_dataloader(
        interactions=dataset.train_set, sessions=dataset.train_session
    )
    for batch in loader:
        assert torch.isfinite(model.training_step(batch, 0))
    model.eval()
    with torch.no_grad():
        scores = model.predict(torch.arange(dataset.info()["n_users"]))
    assert scores.shape == (dataset.info()["n_users"], dataset.info()["n_items"])
    assert torch.isfinite(scores).all()


# Items 2 and 5 carry two genres each, one row per genre: the long layout.
LONG_SIDE = pd.DataFrame(
    {
        "item_id": [2, 2, 3, 5, 5, 7, 8, 9, 10],
        "genre": ["g1", "g2", "g3", "g1", "g4", "g2", "g3", "g4", "g1"],
    }
)


def _build_long() -> Dataset:
    """A dataset whose side table repeats items.

    Returns:
        Dataset: The dataset under test.
    """
    return Dataset(
        train_data=TRAIN,
        eval_data=EVAL,
        side_data=LONG_SIDE,
        timestamp_label="timestamp",
        batch_size=4,
    )


def test_the_feature_matrix_reads_a_side_table_that_repeats_items():
    """The item-by-feature matrix merges every row of an item into that item."""
    dataset = _build_long()
    n_items = dataset.info()["n_items"]
    _, item_map = dataset.get_mappings()

    matrix = dataset.get_feature_matrix()

    assert matrix.shape[0] == n_items + 1
    assert int(matrix[item_map[2]].sum()) == 2
    assert int(matrix[item_map[5]].sum()) == 2


@pytest.mark.parametrize(
    "model_name",
    ["AFM", "DCN", "DCNv2", "DeepFM", "FM", "NFM", "WideAndDeep", "xDeepFM"],
)
def test_context_models_refuse_a_side_table_that_repeats_items(model_name: str):
    """A per-item lookup cannot hold two rows for one item, so the model says so.

    Args:
        model_name (str): The context-aware model under test.
    """
    dataset = _build_long()

    with pytest.raises(ValueError, match="one row per item"):
        model_registry.get(
            model_name,
            params=build_params(model_name),
            info=dataset.info(),
            interactions=dataset.train_set,
            sessions=dataset.train_session,
            transactions=dataset.train_transactions,
        )
