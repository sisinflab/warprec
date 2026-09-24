"""Every registered metric must produce a number from a real evaluation.

This is the metric counterpart of the model smoke suite: deliberately shallow
and wide, making no claim about whether a value is *right* — the golden suites
do that — only that asking for a metric on an ordinary dataset yields a finite
number rather than an exception or a placeholder.

It exists because two metrics used to raise as soon as the inputs they exist to
read were supplied. RSP added catalogue-sized interaction counts into an
accumulator indexed by the cluster lookup, which is a row longer, and SRecall
multiplied the whole feature lookup against a relevance mask over the catalogue
alone. Both were unreachable from any test that did not configure clusters or
item features, so both shipped.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
import torch

from test_metrics_golden import _FixedScoreRecommender, TEST_ROWS, TRAIN_ROWS

import warprec.evaluation  # noqa: F401  (populates the registry)
from warprec.data.dataset import Dataset
from warprec.evaluation import Evaluator
from warprec.evaluation.propensity import build_propensity
from warprec.utils.registry import metric_registry


class _RandomScoreRecommender(_FixedScoreRecommender):
    """Scores drawn once from a fixed seed, for fixtures wider than the shared one.

    Unlike the shared double this one honours a candidate list, because sampled
    evaluation hands the model the items it wants scored rather than expecting
    the whole catalogue back.
    """

    def __init__(self, params: dict, info: dict, **kwargs: Any):
        super().__init__(params, info, **kwargs)
        generator = torch.Generator().manual_seed(17)
        self.scores = torch.rand(
            (info["n_users"], info["n_items"]), generator=generator
        )

    def predict(
        self,
        user_indices: torch.Tensor,
        *args: Any,
        item_indices: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Score the requested users, restricted to the candidates if given.

        Args:
            user_indices (torch.Tensor): The batch of user indices.
            *args (Any): Unused.
            item_indices (Optional[torch.Tensor]): The candidates to score.
            **kwargs (Any): Unused.

        Returns:
            torch.Tensor: The score matrix.
        """
        rows = self.scores[user_indices]
        if item_indices is None:
            return rows
        return rows.gather(1, item_indices.clamp(max=rows.size(1) - 1))


USER_CLUSTERS = pd.DataFrame({"user_id": [0, 1, 2], "cluster": [1, 1, 2]})
ITEM_CLUSTERS = pd.DataFrame({"item_id": [0, 1, 2, 3, 4], "cluster": [1, 1, 1, 2, 2]})
SIDE = pd.DataFrame(
    {"item_id": [0, 1, 2, 3, 4], "a": [1, 1, 0, 0, 0], "b": [0, 0, 1, 1, 1]}
)

# Metrics that combine others and are configured rather than named on their own.
COMPOSITE = {"HYPERVOLUME", "EUCDISTANCE"}

METRICS: List[str] = sorted(
    name for name in metric_registry.list_registered() if name.upper() not in COMPOSITE
)


@pytest.fixture(name="full_dataset", scope="module")
def full_dataset_fixture() -> Dataset:
    """The shared fixture with everything any metric might ask for attached.

    Returns:
        Dataset: The dataset under test.
    """
    columns = ["user_id", "item_id", "rating"]
    return Dataset(
        train_data=pd.DataFrame(TRAIN_ROWS, columns=columns),
        eval_data=pd.DataFrame(TEST_ROWS, columns=columns),
        side_data=SIDE,
        user_cluster=USER_CLUSTERS,
        item_cluster=ITEM_CLUSTERS,
        cluster_label="cluster",
        rating_type="explicit",
        rating_label="rating",
        batch_size=8,
    )


def test_the_discovered_set_is_not_empty():
    """Guards against the discovery above silently matching nothing."""
    assert len(METRICS) > 30, f"only {len(METRICS)} metrics discovered"


@pytest.mark.parametrize("metric_name", METRICS)
def test_a_registered_metric_produces_a_finite_number(
    metric_name: str, full_dataset: Dataset
):
    """Asking for the metric on an ordinary dataset yields a usable value."""
    train_sparse = full_dataset.train_set.get_sparse()

    evaluator = Evaluator(
        [metric_name],
        [3],
        train_set=train_sparse,
        feature_lookup=full_dataset.get_feature_matrix(),
        user_cluster=full_dataset.get_user_cluster(),
        item_cluster=full_dataset.get_item_cluster(),
        propensity=build_propensity(train_sparse.getnnz(axis=0)),
    )
    evaluator.evaluate(
        model=_FixedScoreRecommender({}, full_dataset.info()),
        dataloader=full_dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=full_dataset,
    )
    results: Dict[str, Any] = evaluator.compute_results()[3]

    assert results, f"{metric_name} produced no result at all"

    for key, value in results.items():
        tensor = torch.as_tensor(value, dtype=torch.float)
        assert torch.isfinite(tensor).any(), (
            f"{metric_name} produced no finite value under '{key}'"
        )


@pytest.fixture(name="roomy_dataset", scope="module")
def roomy_dataset_fixture() -> Dataset:
    """A catalogue with room to draw negatives from.

    The shared three-user fixture is deliberately tiny, so every user has seen
    most of the five items and a sampled evaluation cannot draw distinct
    negatives for them. This one is wide enough that it can.

    Returns:
        Dataset: The dataset under test.
    """
    rng = np.random.default_rng(5)
    items = 40
    train_rows, eval_rows = [], []
    for user in range(8):
        seen = rng.choice(items, 6, replace=False)
        train_rows += [(user, int(item), 5.0) for item in seen[:4]]
        eval_rows += [(user, int(item), 5.0) for item in seen[4:]]

    columns = ["user_id", "item_id", "rating"]
    features = pd.DataFrame(
        {
            "item_id": np.arange(items),
            "a": rng.integers(0, 2, items),
            "b": rng.integers(0, 2, items),
        }
    )
    return Dataset(
        train_data=pd.DataFrame(train_rows, columns=columns),
        eval_data=pd.DataFrame(eval_rows, columns=columns),
        side_data=features,
        user_cluster=pd.DataFrame(
            {"user_id": np.arange(8), "cluster": [1, 1, 1, 1, 2, 2, 2, 2]}
        ),
        item_cluster=pd.DataFrame(
            {"item_id": np.arange(items), "cluster": rng.integers(1, 3, items)}
        ),
        cluster_label="cluster",
        rating_type="explicit",
        rating_label="rating",
        batch_size=8,
    )


@pytest.mark.parametrize("metric_name", METRICS)
def test_a_registered_metric_survives_a_sampled_evaluation(
    metric_name: str, roomy_dataset: Dataset
):
    """The same metrics, asked over a sampled candidate set rather than all items.

    Sampled evaluation hands every metric local indices into a candidate list
    plus the mapping back to the catalogue, which is a different branch through
    almost all of them and one no test reached before.
    """
    train_sparse = roomy_dataset.train_set.get_sparse()

    evaluator = Evaluator(
        [metric_name],
        [3],
        train_set=train_sparse,
        feature_lookup=roomy_dataset.get_feature_matrix(),
        user_cluster=roomy_dataset.get_user_cluster(),
        item_cluster=roomy_dataset.get_item_cluster(),
        propensity=build_propensity(train_sparse.getnnz(axis=0)),
    )
    evaluator.evaluate(
        model=_RandomScoreRecommender({}, roomy_dataset.info()),
        dataloader=roomy_dataset.get_sampled_evaluation_dataloader(num_negatives=5),
        strategy="sampled",
        dataset=roomy_dataset,
    )
    results: Dict[str, Any] = evaluator.compute_results()[3]

    assert results, f"{metric_name} produced no result at all under sampling"
    for key, value in results.items():
        tensor = torch.as_tensor(value, dtype=torch.float)
        assert torch.isfinite(tensor).any(), (
            f"{metric_name} produced no finite value under '{key}' when sampling"
        )


def test_asking_for_more_negatives_than_exist_is_refused(full_dataset: Dataset):
    """It used to draw for them forever.

    The fallback that tops up a short candidate list rejected anything the user
    had already seen, with no bound: on a user with fewer unseen items than the
    configuration asks for, that loop never finishes and the run hangs with no
    output at all.
    """
    with pytest.raises(ValueError, match="left unseen"):
        full_dataset.get_sampled_evaluation_dataloader(num_negatives=4)
