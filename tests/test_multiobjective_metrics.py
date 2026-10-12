"""EucDistance and Hypervolume combine any sub-metric, user-averaged or global.

They used to hand every sub-metric the user indices as a second positional
argument, which only a user-averaged metric accepts: a global metric such as
ItemCoverage raised a TypeError on the first batch. A sub-metric that reports
several values, none under its own name, was silently scored zero.
"""

from typing import Any, Dict, List

import pandas as pd
import pytest
import torch

from test_metric_smoke import full_dataset_fixture  # noqa: F401  (a fixture)
from test_metrics_golden import _FixedScoreRecommender, TEST_ROWS, TRAIN_ROWS

import warprec.evaluation  # noqa: F401  (populates the registry)
from warprec.data.dataset import Dataset
from warprec.evaluation import Evaluator
from warprec.utils.config.evaluation_configuration import ComplexMetricConfig

K = 2


@pytest.fixture(name="dataset", scope="module")
def dataset_fixture() -> Dataset:
    """The golden fixture, whose rankings are known exactly.

    Returns:
        Dataset: The dataset under test.
    """
    columns = ["user_id", "item_id", "rating"]
    return Dataset(
        train_data=pd.DataFrame(TRAIN_ROWS, columns=columns),
        eval_data=pd.DataFrame(TEST_ROWS, columns=columns),
        rating_type="explicit",
        rating_label="rating",
        batch_size=8,
    )


def evaluate(dataset: Dataset, complex_metrics: List[Dict[str, Any]]) -> Dict:
    """Evaluate the fixed-score double with Precision, ItemCoverage and the rest.

    Args:
        dataset (Dataset): The dataset to evaluate on.
        complex_metrics (List[Dict[str, Any]]): The composite metrics to add.

    Returns:
        Dict: The results at the cutoff.
    """
    evaluator = Evaluator(
        ["Precision", "ItemCoverage"],
        [K],
        train_set=dataset.train_set.get_sparse(),
        complex_metrics=[ComplexMetricConfig(**c) for c in complex_metrics],
    )
    evaluator.evaluate(
        model=_FixedScoreRecommender({}, dataset.info()),
        dataloader=dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=dataset,
    )
    return evaluator.compute_results()[K]


def only_value(results: Dict, prefix: str) -> Any:
    """Return the one result whose name starts with the prefix.

    Args:
        results (Dict): The results at a cutoff.
        prefix (str): The start of the metric name.

    Returns:
        Any: The value of that result.
    """
    (value,) = [v for name, v in results.items() if name.startswith(prefix)]
    return value


def test_eucdistance_mixes_a_user_averaged_and_a_global_metric(dataset: Dataset):
    """Each user's distance combines their precision with the catalogue's coverage."""
    results = evaluate(
        dataset,
        [
            {
                "name": "EucDistance",
                "params": {
                    "metric_names": ["Precision", "ItemCoverage"],
                    "utopia_points": [1.0, 5.0],
                },
            }
        ],
    )
    precision = results["Precision"]
    coverage = results["ItemCoverage"]

    expected = torch.sqrt((precision - 1.0) ** 2 + (coverage - 5.0) ** 2)
    torch.testing.assert_close(
        only_value(results, "EucDistance"), expected, equal_nan=True
    )


def test_eucdistance_of_global_metrics_alone_is_one_number(dataset: Dataset):
    """With no per-user sub-metric there is no per-user distance to report."""
    results = evaluate(
        dataset,
        [
            {
                "name": "EucDistance",
                "params": {"metric_names": ["ItemCoverage"], "utopia_points": [5.0]},
            }
        ],
    )

    value = only_value(results, "EucDistance")
    assert isinstance(value, float)
    assert value == pytest.approx(abs(results["ItemCoverage"] - 5.0))


def test_hypervolume_mixes_a_user_averaged_and_a_global_metric(dataset: Dataset):
    """Each user's volume is their precision times the catalogue's coverage."""
    results = evaluate(
        dataset,
        [
            {
                "name": "Hypervolume",
                "params": {
                    "metric_names": ["Precision", "ItemCoverage"],
                    "nadir_points": [0.0, 0.0],
                    "higher_is_better": [True, True],
                },
            }
        ],
    )

    expected = results["Precision"] * results["ItemCoverage"]
    torch.testing.assert_close(
        only_value(results, "Hypervolume"), expected, equal_nan=True
    )


def test_hypervolume_of_global_metrics_alone_is_one_number(dataset: Dataset):
    """A global sub-metric alone gives a single volume."""
    results = evaluate(
        dataset,
        [
            {
                "name": "Hypervolume",
                "params": {
                    "metric_names": ["ItemCoverage"],
                    "nadir_points": [1.0],
                    "higher_is_better": [True],
                },
            }
        ],
    )

    value = only_value(results, "Hypervolume")
    assert isinstance(value, float)
    assert value == pytest.approx(results["ItemCoverage"] - 1.0)


@pytest.mark.parametrize(
    "name,params",
    [
        ("EucDistance", {"utopia_points": [0.0, 1.0]}),
        (
            "Hypervolume",
            {"nadir_points": [0.0, 0.0], "higher_is_better": [True, True]},
        ),
    ],
)
def test_a_sub_metric_with_several_values_is_refused(
    name: str, params: Dict[str, Any], full_dataset: Dataset
):
    """BiasDisparityBD reports one value per cluster pair, none under its name.

    A zero used to stand in for it without a word.
    """
    evaluator = Evaluator(
        [],
        [K],
        train_set=full_dataset.train_set.get_sparse(),
        user_cluster=full_dataset.get_user_cluster(),
        item_cluster=full_dataset.get_item_cluster(),
        complex_metrics=[
            ComplexMetricConfig(
                name=name,
                params={"metric_names": ["BiasDisparityBD", "Precision"], **params},
            )
        ],
    )
    evaluator.evaluate(
        model=_FixedScoreRecommender({}, full_dataset.info()),
        dataloader=full_dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=full_dataset,
    )

    with pytest.raises(ValueError, match=f"{name} cannot combine BiasDisparityBD"):
        evaluator.compute_results()
