"""Behavioural tests for the cloze collate function.

Masked-language training produces a different number of targets per sequence, so
the batch has to be padded rather than stacked. Getting the padding value wrong
is not an error: it quietly turns padding into a real item and trains the model
to predict it.
"""

from typing import Tuple

import torch

from warprec.data.entities.train_structures.custom_collate_fn import (
    collate_fn_cloze_mask,
)

PADDING = 99


def sample(
    n_targets: int, seq_len: int = 5, n_negatives: int = 2
) -> Tuple[torch.Tensor, ...]:
    """One cloze example with the given number of masked positions.

    Args:
        n_targets (int): How many positions were masked.
        seq_len (int): The fixed sequence length.
        n_negatives (int): How many negatives per masked position.

    Returns:
        Tuple[torch.Tensor, ...]: The masked sequence, positives, negatives
            and masked indices.
    """
    return (
        torch.arange(seq_len),
        torch.arange(1, n_targets + 1),
        torch.arange(n_targets * n_negatives).view(n_targets, n_negatives),
        torch.arange(n_targets),
    )


def test_sequences_of_equal_length_are_stacked_unchanged():
    """The sequence itself is already fixed-width; it must not be re-padded."""
    batch = [sample(2), sample(2)]

    sequences, positives, negatives, indices = collate_fn_cloze_mask(batch, PADDING)

    assert sequences.shape == (2, 5)
    assert positives.shape == (2, 2)
    assert negatives.shape == (2, 2, 2)
    assert indices.shape == (2, 2)


def test_a_shorter_example_is_padded_up_to_the_longest():
    """The batch is as wide as its widest example, not its first."""
    batch = [sample(1), sample(3)]

    _, positives, negatives, indices = collate_fn_cloze_mask(batch, PADDING)

    assert positives.shape == (2, 3)
    assert negatives.shape == (2, 3, 2)
    assert indices.shape == (2, 3)


def test_padded_targets_carry_the_padding_item():
    """A padded position must be recognisable as one, not as item zero."""
    batch = [sample(1), sample(3)]

    _, positives, negatives, _ = collate_fn_cloze_mask(batch, PADDING)

    # The first example has one real target; the rest of its row is padding.
    assert int(positives[0, 0]) == 1
    assert torch.all(positives[0, 1:] == PADDING)
    assert torch.all(negatives[0, 1:, :] == PADDING)


def test_the_real_values_survive_the_padding():
    """Widening the batch must not disturb what was already there."""
    batch = [sample(1), sample(3)]

    _, positives, negatives, indices = collate_fn_cloze_mask(batch, PADDING)

    assert torch.equal(positives[1], torch.tensor([1, 2, 3]))
    assert torch.equal(negatives[1], torch.arange(6).view(3, 2))
    assert torch.equal(indices[1], torch.tensor([0, 1, 2]))


def test_masked_indices_are_padded_with_zero_not_the_item_padding():
    """They index into the sequence, so the item padding would be out of range."""
    batch = [sample(1, seq_len=5), sample(3, seq_len=5)]

    _, _, _, indices = collate_fn_cloze_mask(batch, PADDING)

    assert torch.all(indices[0, 1:] == 0)
    assert int(indices.max()) < 5
