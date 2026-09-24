"""Metrics that need clusters or item features to be configured at all.

Every metric here was unreachable before: two of them raised on construction or
on the first batch as soon as the inputs they exist to read were supplied, which
is why no run had ever produced a number from them. The cause was the same in
both cases — the cluster and feature lookups carry one row past the catalogue,
for the padding item the model families index when a position holds nothing, and
the metric lined that row up against the catalogue-sized tensors beside it.
"""

from typing import Any, Dict

import pandas as pd
import pytest
import torch

from test_metrics_golden import _FixedScoreRecommender, TEST_ROWS, TRAIN_ROWS

from warprec.data.dataset import Dataset
from warprec.evaluation import Evaluator

# Two user groups and two item groups over the shared fixture:
#   users {0, 1} -> group 1,  user 2 -> group 2
#   items {0, 1, 2} -> group 1,  items {3, 4} -> group 2
USER_CLUSTERS = pd.DataFrame({"user_id": [0, 1, 2], "cluster": [1, 1, 2]})
ITEM_CLUSTERS = pd.DataFrame({"item_id": [0, 1, 2, 3, 4], "cluster": [1, 1, 1, 2, 2]})

# Two features, split so that each item carries exactly one:
#   items 0 and 1 have feature 'a', items 2, 3 and 4 have feature 'b'
SIDE = pd.DataFrame(
    {"item_id": [0, 1, 2, 3, 4], "a": [1, 1, 0, 0, 0], "b": [0, 0, 1, 1, 1]}
)

CLUSTERED = ["REO", "RSP", "PopREO", "PopRSP", "ItemMADRanking", "UserMADRanking"]


@pytest.fixture(name="clustered_dataset", scope="module")
def clustered_dataset_fixture() -> Dataset:
    """The shared fixture with clusters and item features attached.

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


def evaluate(names: list, dataset: Dataset, k: int = 3) -> Dict[str, Any]:
    """Run the given metrics through the real evaluator.

    Args:
        names (list): The registered metric names.
        dataset (Dataset): The fixture.
        k (int): The cutoff to evaluate at.

    Returns:
        Dict[str, Any]: The results at that cutoff.
    """
    evaluator = Evaluator(
        names,
        [k],
        train_set=dataset.train_set.get_sparse(),
        feature_lookup=dataset.get_feature_matrix(),
        user_cluster=dataset.get_user_cluster(),
        item_cluster=dataset.get_item_cluster(),
    )
    evaluator.evaluate(
        model=_FixedScoreRecommender({}, dataset.info()),
        dataloader=dataset.get_evaluation_dataloader(),
        strategy="full",
        dataset=dataset,
    )
    return evaluator.compute_results()[k]


def test_srecall_matches_hand_computed_value(clustered_dataset: Dataset):
    """Subtopic recall is the share of a user's relevant features the list finds.

    user 0 is relevant on items 0 and 3, which between them carry both features;
    its top-3 is 4, 3, 0, whose relevant members are 3 and 0 — both features
    again, so 2/2.
    user 1 is relevant on item 4 alone, carrying feature 'b'; its top-3 is
    1, 4, 2, whose relevant member is 4 — feature 'b', so 1/1.
    """
    scores = evaluate(["SRecall"], clustered_dataset)["SRecall"]

    assert float(scores[0]) == pytest.approx(1.0)
    assert float(scores[1]) == pytest.approx(1.0)
    # The third user has nothing to find, so it is left out of the mean.
    assert torch.isnan(scores[2])
    assert float(scores.nanmean()) == pytest.approx(1.0)


def test_subtopic_recall_falls_when_the_list_misses_a_feature(
    clustered_dataset: Dataset,
):
    """A perfect score has to be earned, so check it can also be lost.

    At a cutoff of two, user 0 sees only items 4 and 3. Making it relevant on
    items 0 and 4 asks for both features — item 0 carries 'a', item 4 carries
    'b' — while the list reaches only item 4, so only 'b' is found.

    Note that a relevant item the user already saw in training does not count:
    the evaluator masks it out of the ground truth as well as out of the
    ranking, which is why this case has to be built from unmasked items.
    """
    columns = ["user_id", "item_id", "rating"]
    narrower = Dataset(
        train_data=pd.DataFrame(TRAIN_ROWS, columns=columns),
        eval_data=pd.DataFrame(
            [(0, 0, 5.0), (0, 4, 5.0), (1, 4, 5.0)], columns=columns
        ),
        side_data=SIDE,
        user_cluster=USER_CLUSTERS,
        item_cluster=ITEM_CLUSTERS,
        cluster_label="cluster",
        rating_type="explicit",
        rating_label="rating",
        batch_size=8,
    )

    scores = evaluate(["SRecall"], narrower, k=2)["SRecall"]

    assert float(scores[0]) == pytest.approx(0.5)


def test_subtopic_recall_reads_the_item_by_feature_matrix(
    clustered_dataset: Dataset,
):
    """It needs the content view of the attributes, not the context one.

    The context encoding gives every attribute column an index per distinct
    value, so a zero means "the first value" rather than "not present" and every
    item holds a non-zero in every column. Fed that, SRecall returned 1.0 for
    every user of every dataset at every cutoff, which is what it did before.
    """
    matrix = clustered_dataset.get_feature_matrix()
    lookup = clustered_dataset.get_features_lookup()

    # The content view has genuine zeros; the context view does not.
    assert float(matrix[:-1].min()) == 0.0
    assert float(lookup[:-1].min()) > 0.0

    # Both carry the padding row every item-indexed lookup carries.
    assert matrix.size(0) == clustered_dataset.info()["n_items"] + 1


@pytest.mark.parametrize("metric_name", CLUSTERED)
def test_a_clustered_metric_produces_a_number(
    metric_name: str, clustered_dataset: Dataset
):
    """Supplying clusters must not break the metrics that require them.

    RSP raised a RuntimeError here before: it added the catalogue's interaction
    counts into an accumulator indexed by the cluster lookup, which is one row
    longer. Any run asking for it with item clusters configured died outright.
    """
    results = evaluate([metric_name], clustered_dataset)

    assert results, f"{metric_name} produced nothing"
    for value in results.values():
        assert torch.isfinite(torch.as_tensor(value)).any(), (
            f"{metric_name} produced no finite value"
        )


def test_srecall_survives_a_full_ranking_with_features(clustered_dataset: Dataset):
    """SRecall raised here before, for the same reason RSP did.

    In full-ranking mode it multiplied the whole feature lookup, padding row
    included, against a relevance mask over the catalogue alone.
    """
    results = evaluate(["SRecall"], clustered_dataset)

    assert "SRecall" in results
    assert torch.isfinite(results["SRecall"]).any()
