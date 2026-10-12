"""The popularity and cluster fairness metrics are NaN where they are undefined.

Each compares the rate at which groups of items are recommended. When a group has
nothing to divide by (no relevant item for the equal-opportunity metrics, nothing
on offer for the statistical-parity ones), or fewer than two groups are left to
compare, there is no rate to compare. PopREO used to return a bare
``tensor(0.0)`` there instead of its dictionary, which made the evaluator's
compute_results fail; the others reported 0.0, which reads as perfectly fair.
"""

import math
from typing import Any, Dict

import pandas as pd
import pytest
import torch

from test_metrics_golden import _FixedScoreRecommender, TEST_ROWS, TRAIN_ROWS

from warprec.data.dataset import Dataset
from warprec.evaluation import Evaluator
from warprec.utils.config.evaluation_configuration import ComplexMetricConfig
from warprec.utils.registry import metric_registry

K = 3
COLUMNS = ["user_id", "item_id", "rating"]

# Popular items 0-3, rare items 4 and 5: short head {0, 1, 2, 3}, long tail {4, 5}.
INTERACTIONS = torch.tensor([4.0, 4.0, 4.0, 6.0, 1.0, 1.0])
# Items 0-3 in cluster 1, items 4 and 5 in cluster 2, the padding row in 0.
CLUSTERS = torch.tensor([1, 1, 1, 1, 2, 2, 0])


def evaluate(dataset: Dataset, complex_metrics: list) -> Dict[str, Any]:
    """Run the given metrics through the real evaluator.

    Args:
        dataset (Dataset): The fixture.
        complex_metrics (list): The metrics, as complex_metrics entries.

    Returns:
        Dict[str, Any]: The results at the cutoff.
    """
    evaluator = Evaluator(
        [],
        [K],
        train_set=dataset.train_set.get_sparse(),
        item_cluster=dataset.get_item_cluster(),
        complex_metrics=[ComplexMetricConfig(**c) for c in complex_metrics],
    )
    evaluator.evaluate(
        model=_FixedScoreRecommender({}, dataset.info()),
        dataloader=dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=dataset,
    )
    return evaluator.compute_results()[K]


def test_an_evaluation_with_an_empty_popularity_group_completes():
    """With every item in the short head, the long tail has nothing to compare.

    A pop_ratio of 0.99 puts the whole catalogue of the fixture in the head.
    """
    dataset = Dataset(
        train_data=pd.DataFrame(TRAIN_ROWS, columns=COLUMNS),
        eval_data=pd.DataFrame(TEST_ROWS, columns=COLUMNS),
        rating_type="explicit",
        rating_label="rating",
        batch_size=8,
    )

    results = evaluate(
        dataset,
        [
            {"name": "PopREO", "params": {"pop_ratio": 0.99}},
            {"name": "PopRSP", "params": {"pop_ratio": 0.99}},
        ],
    )

    assert len(results) == 2
    for value in results.values():
        assert isinstance(value, float) and math.isnan(value)


def test_an_evaluation_with_a_single_cluster_has_nothing_to_compare():
    """With one item cluster there is no second rate for REO or RSP."""
    dataset = Dataset(
        train_data=pd.DataFrame(TRAIN_ROWS, columns=COLUMNS),
        eval_data=pd.DataFrame(TEST_ROWS, columns=COLUMNS),
        item_cluster=pd.DataFrame({"item_id": [0, 1, 2, 3, 4], "cluster": [1] * 5}),
        cluster_label="cluster",
        rating_type="explicit",
        rating_label="rating",
        batch_size=8,
    )

    results = evaluate(dataset, [{"name": "REO"}, {"name": "RSP"}])

    assert math.isnan(results["REO"])
    assert math.isnan(results["RSP"])
    # The one cluster's own rate is still reported.
    assert not math.isnan(results["REO_IC1"])
    assert not math.isnan(results["RSP_IC1"])


def test_popreo_without_relevant_long_tail_items_is_nan():
    """PopREO returns its dictionary, with NaN, rather than a bare tensor."""
    metric = metric_registry.get_class("PopREO")(k=K, item_interactions=INTERACTIONS)
    relevance = torch.zeros(1, 6)
    relevance[0, 0] = 1.0  # one relevant item, in the short head

    metric.update(
        preds=torch.zeros(1, 6),
        binary_relevance=relevance,
        top_3_indices=torch.tensor([[0, 1, 4]]),
        top_3_binary_relevance=torch.tensor([[1.0, 0.0, 0.0]]),
        item_indices=None,
    )
    result = metric.compute()

    assert isinstance(result, dict)
    assert math.isnan(result[metric.name])


def test_poprsp_with_nothing_on_offer_in_a_group_is_nan():
    """Both rare items are out of reach, so the long tail has no rate."""
    metric = metric_registry.get_class("PopRSP")(k=K, item_interactions=INTERACTIONS)
    preds = torch.zeros(1, 6)
    preds[0, 4:] = -torch.inf

    metric.update(
        preds=preds,
        binary_relevance=torch.zeros(1, 6),
        top_3_indices=torch.tensor([[0, 1, 2]]),
        item_indices=None,
    )
    result = metric.compute()

    assert isinstance(result[metric.name], float)
    assert math.isnan(result[metric.name])


def test_reo_with_relevant_items_in_one_cluster_only_is_nan():
    """Cluster 2 holds no relevant item, so only one rate is left."""
    metric = metric_registry.get_class("REO")(k=K, item_cluster=CLUSTERS)
    relevance = torch.zeros(1, 6)
    relevance[0, 0] = 1.0

    metric.update(
        preds=torch.zeros(1, 6),
        binary_relevance=relevance,
        top_3_indices=torch.tensor([[0, 1, 4]]),
        top_3_binary_relevance=torch.tensor([[1.0, 0.0, 0.0]]),
        item_indices=None,
    )
    result = metric.compute()

    assert math.isnan(result["REO"])
    assert result["REO_IC1"] == pytest.approx(1.0)
    assert math.isnan(result["REO_IC2"])


def test_reo_without_any_relevant_item_is_nan():
    """Nothing relevant was evaluated at all."""
    metric = metric_registry.get_class("REO")(k=K, item_cluster=CLUSTERS)

    result = metric.compute()

    assert math.isnan(result["REO"])


def test_rsp_without_any_user_scored_is_nan():
    """No user was scored, so there is no rate for any cluster."""
    metric = metric_registry.get_class("RSP")(
        k=K, num_users=2, item_cluster=CLUSTERS, item_interactions=INTERACTIONS
    )

    result = metric.compute()

    assert math.isnan(result["RSP"])


def test_rsp_with_one_cluster_on_offer_is_nan():
    """Both rare items are out of reach, so cluster 2 has no rate."""
    metric = metric_registry.get_class("RSP")(
        k=K, num_users=2, item_cluster=CLUSTERS, item_interactions=INTERACTIONS
    )
    preds = torch.zeros(1, 6)
    preds[0, 4:] = -torch.inf

    metric.update(
        preds=preds,
        user_indices=torch.tensor([0]),
        binary_relevance=torch.zeros(1, 6),
        top_3_indices=torch.tensor([[0, 1, 2]]),
        valid_users=torch.tensor([1.0]),
        item_indices=None,
    )
    result = metric.compute()

    assert math.isnan(result["RSP"])
    assert result["RSP_IC1"] == pytest.approx(3 / 4)
