"""Ties are broken independently for every user.

A model that scores many items alike - every cold item under a collaborative
model, or every item at all under one that has learned nothing - is ranked by
the tie break. Drawn once per batch, it gave every user of the batch the same
list, so the whole evaluation rested on a handful of draws and moved a lot
from one seed to the next. Drawn per user, the result is an average over
many independent draws, while a ranking with no ties stays exactly the one
``torch.topk`` returns.
"""

import torch

from warprec.data.ranking import top_k_breaking_ties
from warprec.recommenders.reranking.base import Reranker


def _ties(rows: int, items: int, k: int, seed: int) -> torch.Tensor:
    """The top-k indices of an all-tied score matrix.

    Args:
        rows (int): The number of users.
        items (int): The number of items.
        k (int): The cutoff.
        seed (int): The seed of the tie break.

    Returns:
        torch.Tensor: The indices, [rows, k].
    """
    tied = torch.zeros(rows, items)
    return top_k_breaking_ties(tied, k, torch.Generator().manual_seed(seed))[1]


def test_every_user_gets_a_draw_of_their_own():
    """Users with identical, all-tied scores get different lists."""
    indices = _ties(rows=200, items=50, k=5, seed=0)

    distinct_lists = {tuple(row) for row in indices.tolist()}
    assert len(distinct_lists) > 190


def test_every_tied_item_is_equally_likely():
    """Over many users, each tied item reaches the top k about k/n of the time."""
    rows, items, k = 20_000, 40, 4
    indices = _ties(rows=rows, items=items, k=k, seed=1)

    counts = torch.bincount(indices.flatten(), minlength=items).float()
    expected = rows * k / items
    # Binomial sd is about sqrt(2000 * 0.9) ~ 42; allow five of them.
    assert (counts - expected).abs().max() < 5 * 43


def test_a_list_never_repeats_an_item():
    """The draw picks k distinct items for each user."""
    indices = _ties(rows=500, items=30, k=10, seed=2)

    assert all(len(set(row)) == 10 for row in indices.tolist())


def test_rows_without_ties_are_exactly_torch_topk():
    """A row whose ranking is settled is returned bit for bit as torch.topk has it."""
    generator = torch.Generator().manual_seed(5)
    scores = torch.rand(64, 300, generator=generator)
    scores[::2] = 0.0  # every other row is all tied

    values, indices = top_k_breaking_ties(scores, 7, torch.Generator().manual_seed(6))
    expected_values, expected_indices = torch.topk(scores, 7, dim=1)

    torch.testing.assert_close(values[1::2], expected_values[1::2], rtol=0, atol=0)
    torch.testing.assert_close(indices[1::2], expected_indices[1::2], rtol=0, atol=0)


def test_ties_only_reorder_equal_scores():
    """A higher score always ranks first; only equal scores are drawn."""
    # Item 0 scores highest, items 1-3 tie in second place, the rest tie last
    row = torch.tensor([9.0, 5.0, 5.0, 5.0] + [1.0] * 20)
    scores = row.repeat(3_000, 1)

    values, indices = top_k_breaking_ties(scores, 6, torch.Generator().manual_seed(7))

    assert bool((indices[:, 0] == 0).all())
    assert set(indices[:, 1:4].flatten().tolist()) == {1, 2, 3}
    assert bool((indices[:, 4:] >= 4).all())
    torch.testing.assert_close(values, scores.gather(1, indices), rtol=0, atol=0)
    assert bool((values[:, :-1] >= values[:, 1:]).all())

    # Both the order among the tied second places and the choice among the
    # many last places are drawn, so every candidate turns up.
    assert set(indices[:, 1].tolist()) == {1, 2, 3}
    assert set(indices[:, 4:].flatten().tolist()) == set(range(4, 24))


def test_masked_items_stay_below_the_candidates():
    """Items scored -inf are only drawn when too few candidates remain."""
    scores = torch.full((100, 50), -torch.inf)
    scores[:, 10:13] = 0.0

    _, indices = top_k_breaking_ties(scores, 5, torch.Generator().manual_seed(8))

    assert all(set(row[:3]) == {10, 11, 12} for row in indices.tolist())


class _KeepOrder(Reranker):
    """A re-ranker that keeps the pool in the order it was handed over."""

    def _select(
        self,
        relevance: torch.Tensor,
        candidates: torch.Tensor,
        k: int,
        user_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Take the first k of the pool.

        Args:
            relevance (torch.Tensor): The pooled scores.
            candidates (torch.Tensor): The pooled item indices.
            k (int): How many to choose.
            user_indices (torch.Tensor | None): Ignored.

        Returns:
            torch.Tensor: The first k positions.
        """
        return torch.arange(k).repeat(relevance.size(0), 1)


def test_the_reranker_pool_breaks_ties_like_the_ranking():
    """The pool a re-ranker reconsiders is drawn per user, not taken by item id."""
    reranker = _KeepOrder(item_features=torch.zeros(200, 1), pool=10)
    tied = torch.zeros(50, 200)

    _, indices = reranker(tied, 5, generator=torch.Generator().manual_seed(9))
    _, positional = reranker(tied, 5)

    assert bool((positional < 10).all()), "without a generator the pool is positional"
    assert int(indices.max()) >= 10, "the pool stayed at the lowest ids"
    assert len({tuple(row) for row in indices.tolist()}) > 45


def test_a_tied_pool_is_drawn_from_whole_and_only_from_itself():
    """A pool tied apart from a masked catalogue, as under candidates: cold."""
    scores = torch.full((2_000, 400), -torch.inf)
    pool = torch.arange(100, 400, 15)  # 20 items spread over the catalogue
    scores[:, pool] = 0.0

    values, indices = top_k_breaking_ties(scores, 5, torch.Generator().manual_seed(10))

    assert set(indices.flatten().tolist()) == set(pool.tolist())
    assert bool((values == 0.0).all())
    assert len({tuple(row) for row in indices.tolist()}) > 1_900


def test_a_users_draw_does_not_depend_on_their_batch():
    """A user is ranked alike whoever shares the batch, so lists can be compared.

    The evaluator and the writer walk the users in different batches; only a
    draw keyed on the user and the item gives both the same list.
    """
    scores = torch.zeros(60, 80)
    scores[:, 70:] = 1.0  # some ties above the cutoff too
    users = torch.arange(100, 160)

    whole = top_k_breaking_ties(scores, 12, torch.Generator().manual_seed(4), users)
    first = top_k_breaking_ties(
        scores[:25], 12, torch.Generator().manual_seed(4), users[:25]
    )
    shuffled = torch.randperm(35, generator=torch.Generator().manual_seed(0)) + 25
    rest = top_k_breaking_ties(
        scores[shuffled], 12, torch.Generator().manual_seed(4), users[shuffled]
    )

    torch.testing.assert_close(whole[1][:25], first[1], rtol=0, atol=0)
    torch.testing.assert_close(whole[1][shuffled], rest[1], rtol=0, atol=0)


def test_tied_items_come_first_for_about_half_the_users():
    """For any two tied items, either one leads for about half of the users."""
    scores = torch.zeros(20_000, 30)
    _, indices = top_k_breaking_ties(
        scores, 30, torch.Generator().manual_seed(5), torch.arange(20_000)
    )
    position = torch.argsort(indices, dim=1)  # where each item landed, per user

    for a, b in [(0, 1), (3, 17), (28, 29)]:
        share = float((position[:, a] < position[:, b]).float().mean())
        assert abs(share - 0.5) < 0.02, (a, b, share)
