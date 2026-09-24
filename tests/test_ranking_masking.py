"""How a run decides which items a user is still allowed to be recommended.

Masking is subtractive and silent: an item wrongly excluded never appears, and
nothing in the output says why. The contextual variant is the subtle one — an
item watched on a weekday morning is a fair recommendation for a Saturday night,
so it must mask per context rather than per pair.
"""

from typing import Dict, Tuple

import numpy as np
import torch

from warprec.data.ranking import (
    mask_seen_in_context,
    restrict_to_candidates,
    top_k_breaking_ties,
)


def test_only_the_context_the_user_was_seen_in_is_masked():
    """The same item stays recommendable in a context the user has not had it in."""
    predictions = torch.zeros((2, 4))
    # Row 0 is user 0 in context A, row 1 is user 0 in context B.
    context_rows = np.array([[1.0], [2.0]])
    context_ids: Dict[tuple, int] = {(1.0,): 10, (2.0,): 20}
    # The user saw item 2 in context A only.
    context_index: Dict[Tuple[int, int], np.ndarray] = {(0, 10): np.array([2])}

    mask_seen_in_context(
        predictions, torch.tensor([0, 0]), context_rows, context_index, context_ids
    )

    assert predictions[0, 2] == -torch.inf, "the seen context was not masked"
    assert predictions[1, 2] == 0.0, "an unseen context was masked anyway"


def test_a_context_never_seen_in_training_masks_nothing():
    """There is nothing to exclude, so the row has to be left alone."""
    predictions = torch.zeros((1, 3))
    context_rows = np.array([[9.0]])

    masked = mask_seen_in_context(predictions, torch.tensor([0]), context_rows, {}, {})

    assert masked == 0
    assert torch.all(predictions == 0.0)


def test_a_repeated_ground_truth_is_counted():
    """A run needs to know how often the answer was something it just masked away."""
    predictions = torch.zeros((2, 4))
    context_rows = np.array([[1.0], [1.0]])
    context_ids: Dict[tuple, int] = {(1.0,): 10}
    context_index: Dict[Tuple[int, int], np.ndarray] = {
        (0, 10): np.array([2]),
        (1, 10): np.array([3]),
    }

    repeated = mask_seen_in_context(
        predictions,
        torch.tensor([0, 1]),
        context_rows,
        context_index,
        context_ids,
        target_items=torch.tensor([2, 0]),
    )

    # The first row's answer was masked, the second row's was not.
    assert repeated == 1


def test_restricting_to_candidates_leaves_only_those_items():
    """Everything outside the candidate set has to become unreachable."""
    predictions = torch.ones((2, 5))
    # The candidate set is a boolean mask over the catalogue, True to keep.
    keep = torch.tensor([False, True, False, True, False])

    restrict_to_candidates(predictions, keep)

    for column in (0, 2, 4):
        assert torch.all(predictions[:, column] == -torch.inf)
    for column in (1, 3):
        assert torch.all(predictions[:, column] == 1.0)


def test_ties_are_broken_by_chance_rather_than_by_item_id():
    """`torch.topk` settles equal scores by position, which is not neutral.

    A model that scores a whole population alike would otherwise always return
    the same items, which reads as a systematic preference the model does not
    have. Asserting that one seed differs from the deterministic answer would
    be a coin toss, so the property is checked across seeds instead: over a
    spread of them the selection has to actually move.
    """
    # Every item scores the same, so the entire ranking is a tie.
    predictions = torch.zeros((1, 8))

    selections = {
        tuple(
            top_k_breaking_ties(
                predictions.clone(), 3, torch.Generator().manual_seed(seed)
            )[1][0].tolist()
        )
        for seed in range(12)
    }

    assert len(selections) > 1, "the tie was settled the same way every time"
    # And more than three of the eight items were reachable across the seeds,
    # so it is not merely permuting one fixed prefix.
    assert len({item for selection in selections for item in selection}) > 3


def test_the_tie_break_is_reproducible_for_a_given_seed():
    """Chance is not the same as irreproducibility: a run must repeat exactly."""
    predictions = torch.zeros((1, 8))

    first = top_k_breaking_ties(
        predictions.clone(), 3, torch.Generator().manual_seed(4)
    )[1]
    again = top_k_breaking_ties(
        predictions.clone(), 3, torch.Generator().manual_seed(4)
    )[1]

    assert torch.equal(first, again)


def test_an_unambiguous_ranking_is_left_exactly_as_it_is():
    """Only rows whose order is genuinely in doubt pay the shuffling cost."""
    predictions = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5]])

    shuffled = top_k_breaking_ties(
        predictions.clone(), 3, torch.Generator().manual_seed(7)
    )
    plain = torch.topk(predictions, 3, dim=1)

    assert torch.equal(shuffled[1], plain.indices)
    assert torch.equal(shuffled[0], plain.values)


def test_without_a_generator_the_ordering_is_the_deterministic_one():
    """The fallback has to match `torch.topk` exactly, ties included.

    Which items `torch.topk` returns for a tie is a property of the kernel, not
    a promise of the API, and it differs between platforms. The contract here is
    only that the fallback defers to it, so that is what is asserted rather than
    any particular ordering.
    """
    predictions = torch.zeros((1, 6))

    values, indices = top_k_breaking_ties(predictions, 3)
    expected = torch.topk(predictions, 3, dim=1)

    assert torch.equal(indices, expected.indices)
    assert torch.equal(values, expected.values)

    # And it is stable: without a generator there is nothing to vary.
    assert torch.equal(indices, top_k_breaking_ties(predictions, 3)[1])
