"""Metrics are pinned against values worked out by hand.

A metric that silently changes is worse than one that crashes: every number the
framework publishes rests on these. The expected values below are derived in the
comments rather than captured from a previous run, so a regression cannot be
rationalised by updating the fixture.
"""

import math
from typing import Any, Dict, Optional

import pandas as pd
import pytest
import torch
from torch import Tensor

from warprec.data.dataset import Dataset
from warprec.evaluation import Evaluator
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.registry import metric_registry

# --------------------------------------------------------------------------
# Ranking metrics, driven through the real Evaluator
# --------------------------------------------------------------------------
#
# Three users, five items. Training pins items so that the evaluator masks them,
# which leaves each user with a known candidate ranking:
#
#   user 0  trained on {1, 2}   relevant {0, 3}   ranking after masking: 4, 3, 0
#   user 1  trained on {0, 3}   relevant {4}      ranking after masking: 1, 4, 2
#   user 2  trained on {4, 0}   no test items     excluded from the metrics
#
# user 0: hits at ranks 2 and 3   DCG = 1/log2(3) + 1/log2(4),  IDCG = 1/log2(2) + 1/log2(3)
# user 1: hit at rank 2           DCG = 1/log2(3),              IDCG = 1/log2(2)
U0_NDCG = (1 / math.log2(3) + 1 / math.log2(4)) / (1 / math.log2(2) + 1 / math.log2(3))
U1_NDCG = (1 / math.log2(3)) / (1 / math.log2(2))

EXPECTED_AT_3: Dict[str, float] = {
    "nDCG": (U0_NDCG + U1_NDCG) / 2,
    "Recall": (2 / 2 + 1 / 1) / 2,
    "Precision": (2 / 3 + 1 / 3) / 2,
    "HitRate": (1.0 + 1.0) / 2,
    "MRR": (1 / 2 + 1 / 2) / 2,
    "MAP": ((1 / 2 + 2 / 3) / 2 + (1 / 2) / 1) / 2,
}

TRAIN_ROWS = [
    (0, 1, 5.0),
    (0, 2, 5.0),
    (1, 0, 5.0),
    (1, 3, 5.0),
    (2, 4, 5.0),
    (2, 0, 5.0),
]
TEST_ROWS = [(0, 0, 5.0), (0, 3, 5.0), (1, 4, 5.0)]

# Scores are fixed, so the ranking is known exactly. Masked items may hold any
# value; the evaluator overwrites them.
SCORES = torch.tensor(
    [
        [0.7, 0.0, 0.0, 0.8, 0.9],  # user 0 -> 4, 3, 0
        [0.0, 0.9, 0.7, 0.0, 0.8],  # user 1 -> 1, 4, 2
        [0.5, 0.4, 0.3, 0.2, 0.1],  # user 2 -> excluded, no relevant items
    ]
)


class _FixedScoreRecommender(Recommender):
    """A recommender that returns a score matrix decided by the test."""

    def __init__(self, params: dict, info: dict, **kwargs: Any):
        super().__init__(params, info, **kwargs)
        self.scores = SCORES

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Return the fixed scores for the requested users.

        Args:
            user_indices (Tensor): The batch of user indices.
            *args (Any): Unused.
            item_indices (Optional[Tensor]): Unused; the fixture is full-ranking.
            **kwargs (Any): Unused.

        Returns:
            Tensor: The score matrix for those users.
        """
        return self.scores[user_indices]

    def forward(self, *args: Any, **kwargs: Any):
        """Unused by the evaluation path."""
        raise NotImplementedError


@pytest.fixture(name="golden_dataset")
def golden_dataset_fixture() -> Dataset:
    """The three-user fixture the expected values above are derived from.

    Returns:
        Dataset: The dataset under test.
    """
    columns = ["user_id", "item_id", "rating"]
    train = pd.DataFrame(TRAIN_ROWS, columns=columns)
    evaluation = pd.DataFrame(TEST_ROWS, columns=columns)
    return Dataset(
        train_data=train,
        eval_data=evaluation,
        rating_type="explicit",
        rating_label="rating",
        batch_size=8,
    )


@pytest.mark.parametrize("metric_name,expected", sorted(EXPECTED_AT_3.items()))
def test_ranking_metric_matches_hand_computed_value(
    metric_name: str, expected: float, golden_dataset: Dataset
):
    """Each ranking metric reproduces the value derived in the header."""
    evaluator = Evaluator(
        [metric_name], [3], train_set=golden_dataset.train_set.get_sparse()
    )
    model = _FixedScoreRecommender({}, golden_dataset.info())

    evaluator.evaluate(
        model=model,
        dataloader=golden_dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=golden_dataset,
    )
    results = evaluator.compute_results()

    # The evaluator returns a value per user, with NaN for the users that have
    # no relevant item; the framework reports the nanmean of that (std_logs.py
    # and base_writer.py both do), so the test pins what a user actually sees.
    got = float(results[3][metric_name].nanmean())
    assert got == pytest.approx(expected, abs=1e-6), (
        f"{metric_name}@3 is {got}, expected {expected}"
    )


# --------------------------------------------------------------------------
# Rating metrics, computed directly
# --------------------------------------------------------------------------
#
# Predicted 2, 4, 6 against a ground truth of 1, 4, 9; the third item is unrated
# for the second user, so only the rated cells count.
#   errors: |2-1| = 1, |4-4| = 0, |6-9| = 3   ->  MAE = 4/3
#   squares:   1,        0,         9         ->  MSE = 10/3, RMSE = sqrt(10/3)
RATING_PREDS = torch.tensor([[2.0, 4.0, 6.0]])
RATING_TRUTH = torch.tensor([[1.0, 4.0, 9.0]])
EXPECTED_RATING = {
    "MAE": 4 / 3,
    "MSE": 10 / 3,
    "RMSE": math.sqrt(10 / 3),
}


@pytest.mark.parametrize("metric_name,expected", sorted(EXPECTED_RATING.items()))
def test_rating_metric_matches_hand_computed_value(metric_name: str, expected: float):
    """Each rating metric reproduces the error worked out in the header."""
    metric = metric_registry.get_class(metric_name)(k=3, num_users=1)
    metric.update(
        preds=RATING_PREDS,
        user_indices=torch.tensor([0]),
        ground=RATING_TRUTH,
    )
    got = float(metric.compute()[metric_name])
    assert got == pytest.approx(expected, abs=1e-6), (
        f"{metric_name} is {got}, expected {expected}"
    )
