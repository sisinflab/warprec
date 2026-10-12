"""Sampled evaluation grades the positives the way full evaluation does.

On explicit data full evaluation scores each test item with its rating, so
nDCG is graded; sampled evaluation used to set every positive to 1. With every
unseen item drawn as a negative, the sampled candidates are exactly the items
full evaluation ranks, so the two strategies must then agree user by user.
"""

from typing import Dict

import pandas as pd
import pytest
import torch

from warprec.data.dataset import Dataset
from warprec.evaluation import Evaluator
from warprec.utils.enums import RatingType
from warprec.utils.registry import model_registry

from conftest import build_params

N_USERS = 6
N_ITEMS = 12
TRAIN_PER_USER = 3
EVAL_RATINGS = [5.0, 1.0, 3.0]
METRICS = ["nDCG", "Precision", "Recall", "HitRate", "MAP", "MRR"]
CUTOFFS = [2, 5]


def _dataset(rating_type: str) -> Dataset:
    """Every user has three training items and three rated test items.

    Args:
        rating_type (str): 'explicit' or 'implicit'.

    Returns:
        Dataset: The dataset under test.
    """
    train_rows, eval_rows = [], []
    for user in range(N_USERS):
        items = [(user * 5 + offset) % N_ITEMS for offset in range(6)]
        for position, item in enumerate(items[:TRAIN_PER_USER]):
            train_rows.append((user, item, 4.0, position))
        for rating, item in zip(EVAL_RATINGS, items[TRAIN_PER_USER:]):
            eval_rows.append((user, item, rating, 10))
    columns = ["user_id", "item_id", "rating", "timestamp"]
    return Dataset(
        train_data=pd.DataFrame(train_rows, columns=columns),
        eval_data=pd.DataFrame(eval_rows, columns=columns),
        rating_type=RatingType(rating_type),
        rating_label="rating",
        timestamp_label="timestamp",
        batch_size=4,
    )


def _results(dataset: Dataset, strategy: str) -> Dict[int, Dict[str, torch.Tensor]]:
    """Evaluate an untrained BPR, whose random scores hold no ties.

    Args:
        dataset (Dataset): The dataset to evaluate on.
        strategy (str): 'full' or 'sampled'.

    Returns:
        Dict[int, Dict[str, torch.Tensor]]: The per-user results per cutoff.
    """
    torch.manual_seed(0)
    model = model_registry.get("BPR", params=build_params("BPR"), info=dataset.info())
    evaluator = Evaluator(METRICS, CUTOFFS, train_set=dataset.train_set.get_sparse())
    if strategy == "full":
        loader = dataset.get_evaluation_dataloader()
    else:
        # Every item the user has not seen: the full candidate set.
        unseen = N_ITEMS - TRAIN_PER_USER - len(EVAL_RATINGS)
        loader = dataset.get_sampled_evaluation_dataloader(num_negatives=unseen)
    evaluator.evaluate(model, loader, strategy, dataset)
    return evaluator.compute_results()  # type: ignore[return-value]


@pytest.mark.parametrize("rating_type", ["explicit", "implicit"])
def test_sampled_over_every_unseen_item_matches_full(rating_type: str):
    """With the whole catalogue as candidates, sampled equals full per user.

    Args:
        rating_type (str): 'explicit' or 'implicit'.
    """
    dataset = _dataset(rating_type)
    full = _results(dataset, "full")
    sampled = _results(dataset, "sampled")
    for k in CUTOFFS:
        for name in METRICS:
            torch.testing.assert_close(
                sampled[k][name], full[k][name], msg=f"{name}@{k} ({rating_type})"
            )


def test_sampled_batches_carry_the_ratings_of_the_positives():
    """Each positive arrives with the value the evaluation matrix holds for it."""
    dataset = _dataset("explicit")
    eval_sparse = dataset.eval_set.get_sparse()
    loader = dataset.get_sampled_evaluation_dataloader(num_negatives=2)
    for users, positives, _negatives, values in loader:
        for row, user in enumerate(users.tolist()):
            expected = torch.tensor(
                eval_sparse[user, positives[row].tolist()].toarray().ravel(),
                dtype=torch.float32,
            )
            torch.testing.assert_close(values[row], expected)
            assert sorted(values[row].tolist()) == sorted(EVAL_RATINGS)
