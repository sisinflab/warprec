"""Behavioural tests for the debiased estimators.

The anchor every one of these rests on is that switching the correction off has
to give the naive metric back exactly. Anything else means the weighting is doing
something other than what it claims, and a debiased number is only worth reporting
if that reduction holds.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch
from torch import Tensor

from warprec.data.dataset import Dataset
from warprec.evaluation import build_evaluator
from warprec.evaluation.propensity import build_propensity
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.config import EvaluationConfig

FLAT = {"estimator": "popularity", "power": 0.0, "clip": 0.1}
CORRECTED = {"estimator": "popularity", "power": 0.5, "clip": 0.1}


class FixedScores(Recommender):
    """A recommender that returns a score matrix decided by the test."""

    scores: Tensor

    def predict(
        self,
        user_indices: Tensor,
        *args: Any,
        item_indices: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Return the fixed scores of the requested users.

        Args:
            user_indices (Tensor): The batch of users.
            *args (Any): Unused.
            item_indices (Optional[Tensor]): Unused; the fixture ranks fully.
            **kwargs (Any): Unused.

        Returns:
            Tensor: The score matrix of those users.
        """
        return self.scores[user_indices]

    def forward(self, *args: Any, **kwargs: Any):
        """Unused by the evaluation path.

        Args:
            *args (Any): Unused.
            **kwargs (Any): Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


def scored(dataset: Dataset, scores: Tensor) -> FixedScores:
    """Wrap a score matrix in a recommender.

    Args:
        dataset (Dataset): The dataset under test.
        scores (Tensor): The scores to return.

    Returns:
        FixedScores: The recommender.
    """
    model = FixedScores({}, dataset.info())
    model.scores = scores
    return model


def run(
    dataset: Dataset, model: Recommender, metrics: List[str], propensity: Dict
) -> Dict:
    """Evaluate one model and return the mean of each metric.

    Args:
        dataset (Dataset): The dataset under test.
        model (Recommender): The model to evaluate.
        metrics (List[str]): The metrics to compute.
        propensity (Dict): The propensity configuration.

    Returns:
        Dict: The mean of each metric at k=5.
    """
    config = EvaluationConfig(
        top_k=[5],
        metrics=metrics,
        validation_metric="nDCG@5",
        batch_size=32,
        propensity=propensity,
    )
    evaluator = build_evaluator(config, dataset)
    evaluator.evaluate(
        model=model,
        dataloader=dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=dataset,
    )
    return {n: float(v.nanmean()) for n, v in evaluator.compute_results()[5].items()}


def test_popularity_propensity_follows_the_counts():
    """A more popular item carries a higher propensity, floored at the clip."""
    propensity = build_propensity(np.array([100, 50, 10, 1, 0]), power=0.5, clip=0.1)

    assert propensity is not None
    assert float(propensity[0]) == pytest.approx(1.0)
    assert propensity[0] > propensity[1] > propensity[2]
    assert float(propensity.min()) == pytest.approx(0.1)
    assert float(propensity.max()) <= 1.0


def test_a_uniform_estimator_makes_no_correction():
    """'uniform' yields no propensities at all, rather than a vector of ones."""
    assert build_propensity(np.array([5, 1]), estimator="uniform") is None


def test_an_untouched_catalogue_carries_no_signal():
    """With no interactions anywhere, every item is equally observable."""
    propensity = build_propensity(np.zeros(4))

    assert propensity is not None
    torch.testing.assert_close(propensity, torch.ones(4))


def test_an_unknown_estimator_is_refused():
    """A misspelled estimator raises rather than silently not correcting."""
    with pytest.raises(ValueError, match="not supported"):
        build_propensity(np.array([1, 2]), estimator="inverse-vibes")


@pytest.mark.parametrize(
    "debiased,naive", [("IPSRecall", "Recall"), ("SNIPSRecall", "Recall")]
)
def test_a_flat_propensity_gives_the_naive_metric_back(
    dataset: Dataset, debiased: str, naive: str
):
    """With every propensity equal to one there is nothing left to correct."""
    torch.manual_seed(0)
    model = scored(
        dataset, torch.rand(dataset.info()["n_users"], dataset.info()["n_items"])
    )

    got = run(dataset, model, [naive, debiased], FLAT)

    assert got[debiased] == pytest.approx(got[naive], abs=1e-6)


def test_the_self_normalised_estimator_stays_bounded(dataset: Dataset):
    """SNIPS divides by the weights it used, which bounds it by construction."""
    torch.manual_seed(0)
    model = scored(
        dataset, torch.rand(dataset.info()["n_users"], dataset.info()["n_items"])
    )

    got = run(dataset, model, ["SNIPSRecall"], CORRECTED)

    assert 0.0 <= got["SNIPSRecall"] <= 1.0


def test_a_debiased_metric_without_propensities_is_refused(dataset: Dataset):
    """A corrected name must never be attached to an uncorrected number."""
    torch.manual_seed(0)
    model = scored(
        dataset, torch.rand(dataset.info()["n_users"], dataset.info()["n_items"])
    )

    with pytest.raises(ValueError, match="propensit"):
        run(dataset, model, ["IPSRecall"], {"estimator": "uniform"})


def test_the_correction_costs_a_popularity_ranker_its_advantage(dataset: Dataset):
    """Ranking by popularity must lose ground once exposure is accounted for.

    This is the property the estimators exist for: the naive metric rewards a
    model for reproducing whatever the log already favoured.
    """
    counts = torch.as_tensor(dataset.train_set.get_sparse().getnnz(axis=0)).float()
    n_users = dataset.info()["n_users"]

    by_popularity = scored(dataset, counts.unsqueeze(0).expand(n_users, -1).clone())
    torch.manual_seed(0)
    by_chance = scored(dataset, torch.rand(n_users, dataset.info()["n_items"]))

    naive_gap = (
        run(dataset, by_popularity, ["Recall"], FLAT)["Recall"]
        / run(dataset, by_chance, ["Recall"], FLAT)["Recall"]
    )
    debiased_gap = (
        run(dataset, by_popularity, ["IPSRecall"], CORRECTED)["IPSRecall"]
        / run(dataset, by_chance, ["IPSRecall"], CORRECTED)["IPSRecall"]
    )

    assert debiased_gap < naive_gap
