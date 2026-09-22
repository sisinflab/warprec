"""Behavioural tests for the re-rankers.

Each re-ranker optimises one thing, so each is asked whether it improved that
thing and left the other one alone. A re-ranker that merely shuffled would move
both, or neither, and would pass a test that only checked the lists had changed.
"""

from typing import Tuple

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix
from torch import Tensor

from warprec.data.dataset import Dataset
from warprec.recommenders.reranking import Calibration, MMR, build_reranker
from warprec.utils.config import RerankConfig

N_USERS, N_ITEMS, N_FEATURES = 24, 40, 6


@pytest.fixture(scope="module")
def features() -> Tensor:
    """Item attributes with clear groups, so redundancy is measurable.

    Returns:
        Tensor: The {item x feature} matrix.
    """
    rng = np.random.default_rng(5)
    matrix = np.zeros((N_ITEMS, N_FEATURES))
    for item in range(N_ITEMS):
        matrix[item, item % N_FEATURES] = 1.0
        if rng.random() < 0.3:
            matrix[item, rng.integers(0, N_FEATURES)] = 1.0
    return torch.as_tensor(matrix, dtype=torch.float)


@pytest.fixture(scope="module")
def history() -> csr_matrix:
    """A training history with a lopsided taste per user.

    Returns:
        csr_matrix: The {user x item} matrix.
    """
    rng = np.random.default_rng(6)
    dense = np.zeros((N_USERS, N_ITEMS))
    for user in range(N_USERS):
        for item in rng.choice(N_ITEMS, 8, replace=False):
            dense[user, item] = 1.0
    return csr_matrix(dense)


@pytest.fixture(scope="module")
def scores() -> Tensor:
    """A score matrix whose head is deliberately redundant.

    Returns:
        Tensor: The {user x item} scores.
    """
    rng = np.random.default_rng(7)
    return torch.as_tensor(rng.random((N_USERS, N_ITEMS)), dtype=torch.float)


def intra_list_similarity(lists: Tensor, features: Tensor) -> float:
    """How alike the items of a list are, averaged over the lists.

    Args:
        lists (Tensor): The item indices of each list.
        features (Tensor): The item attributes.

    Returns:
        float: The mean off-diagonal cosine similarity.
    """
    picked = features[lists]
    unit = picked / picked.norm(dim=2, keepdim=True).clamp(min=1e-12)
    similarity = torch.bmm(unit, unit.transpose(1, 2))
    k = lists.size(1)
    off = similarity.sum(dim=(1, 2)) - similarity.diagonal(dim1=1, dim2=2).sum(1)
    return float((off / (k * (k - 1))).mean())


def calibration_error(lists: Tensor, features: Tensor, history: csr_matrix) -> float:
    """How far a list's make-up sits from the user's own.

    Args:
        lists (Tensor): The item indices of each list.
        features (Tensor): The item attributes.
        history (csr_matrix): The training interactions.

    Returns:
        float: The mean divergence.
    """
    mix = features / features.sum(dim=1, keepdim=True).clamp(min=1e-12)
    per_user = np.asarray(history.sum(axis=1)).ravel()
    target = torch.as_tensor(
        (history @ mix.numpy()) / np.clip(per_user, 1e-12, None)[:, None],
        dtype=torch.float,
    )
    listed = mix[lists].mean(dim=1)
    smoothed = 0.99 * listed + 0.01 * target
    return float(
        (target * (target.clamp(min=1e-12) / smoothed.clamp(min=1e-12)).log())
        .sum(dim=1)
        .mean()
    )


def baseline(scores: Tensor, k: int = 8) -> Tuple[Tensor, Tensor]:
    """The ranking a run produces without any re-ranker.

    Args:
        scores (Tensor): The score matrix.
        k (int): The cutoff.

    Returns:
        Tuple[Tensor, Tensor]: The scores and item indices.
    """
    return torch.topk(scores, k, dim=1)


def test_mmr_makes_a_list_less_redundant(features: Tensor, scores: Tensor):
    """Cutting redundancy is the whole of MMR's objective."""
    _, plain = baseline(scores)
    _, reranked = MMR(features, pool=20, diversity=0.7)(scores, 8)

    assert intra_list_similarity(reranked, features) < intra_list_similarity(
        plain, features
    )


def test_mmr_diversifies_further_the_more_it_is_asked_to(
    features: Tensor, scores: Tensor
):
    """The weight has to be worth tuning, so it must actually govern the trade."""
    _, mild = MMR(features, pool=20, diversity=0.3)(scores, 8)
    _, strong = MMR(features, pool=20, diversity=0.9)(scores, 8)

    assert intra_list_similarity(strong, features) < intra_list_similarity(
        mild, features
    )


def test_mmr_at_zero_diversity_keeps_the_model_ordering(
    features: Tensor, scores: Tensor
):
    """With no weight on redundancy there is nothing to trade against relevance."""
    _, plain = baseline(scores)
    _, reranked = MMR(features, pool=20, diversity=0.0)(scores, 8)

    torch.testing.assert_close(reranked, plain)


def test_calibration_brings_a_list_closer_to_the_user(
    features: Tensor, scores: Tensor, history: csr_matrix
):
    """Matching the user's own make-up is the whole of calibration's objective."""
    _, plain = baseline(scores)
    reranker = Calibration(features, pool=20, weight=0.8, user_history=history)
    _, reranked = reranker(scores, 8, torch.arange(N_USERS))

    assert calibration_error(reranked, features, history) < calibration_error(
        plain, features, history
    )


def test_each_reranker_is_better_at_its_own_objective(
    features: Tensor, scores: Tensor, history: csr_matrix
):
    """The two must not be interchangeable, or neither is doing what it claims."""
    _, diversified = MMR(features, pool=20, diversity=0.8)(scores, 8)
    _, calibrated = Calibration(features, pool=20, weight=0.8, user_history=history)(
        scores, 8, torch.arange(N_USERS)
    )

    assert intra_list_similarity(diversified, features) < intra_list_similarity(
        calibrated, features
    )
    assert calibration_error(calibrated, features, history) < calibration_error(
        diversified, features, history
    )


def test_the_scores_that_come_back_belong_to_the_items(
    features: Tensor, scores: Tensor
):
    """A re-ranker reorders a list; it does not invent the numbers on it."""
    values, indices = MMR(features, pool=20, diversity=0.5)(scores, 8)

    torch.testing.assert_close(values, scores.gather(1, indices))


def test_a_cutoff_deeper_than_the_pool_still_returns_k_items(
    features: Tensor, scores: Tensor
):
    """Asking for more than was reconsidered must not lose the remainder."""
    values, indices = MMR(features, pool=5, diversity=0.5)(scores, 12)

    assert indices.shape == (N_USERS, 12)
    assert values.shape == (N_USERS, 12)
    for row in indices:
        assert len(set(row.tolist())) == 12, "an item was returned twice"


@pytest.mark.parametrize("diversity", [-0.1, 1.5])
def test_a_weight_outside_the_unit_interval_is_refused(
    features: Tensor, diversity: float
):
    """The weight is a proportion, so a value outside it is a configuration error."""
    with pytest.raises(ValueError, match="diversity"):
        MMR(features, pool=10, diversity=diversity)


def test_a_pool_of_nothing_is_refused(features: Tensor):
    """A pool has to hold something for there to be anything to reorder."""
    with pytest.raises(ValueError, match="pool"):
        MMR(features, pool=0)


def test_calibration_without_a_history_is_refused(features: Tensor):
    """There is no make-up to match without the interactions to read it from."""
    with pytest.raises(ValueError, match="history|interactions"):
        Calibration(features, pool=10, weight=0.5)


def test_a_reranker_without_side_information_is_refused(dataset: Dataset):
    """Both objectives order items by what they are, so attributes are required."""
    stripped = Dataset(
        train_data=dataset.train_set.get_df(),
        eval_data=dataset.eval_set.get_df(),
        rating_type="implicit",
        batch_size=16,
    )

    with pytest.raises(ValueError, match="side information"):
        build_reranker(RerankConfig(name="MMR"), stripped)


def test_no_reranker_is_built_when_none_is_configured(dataset: Dataset):
    """An absent section leaves the ranking to the model, as before."""
    assert build_reranker(RerankConfig(), dataset) is None


def test_an_unknown_reranker_is_refused():
    """A misspelled name must fail when the configuration is read."""
    with pytest.raises(ValueError, match="not found in registry"):
        RerankConfig(name="MostlyHarmless")
