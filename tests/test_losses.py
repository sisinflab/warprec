"""The pairwise losses compare what they are meant to compare."""

import torch
from torch.nn import functional as F

from warprec.recommenders.losses import BPRLoss


def test_bpr_contrasts_each_positive_with_its_own_negative():
    """One negative per row is that row's negative, not every row's."""
    positive = torch.tensor([1.0, 2.0, 3.0])
    negative = torch.tensor([0.0, 5.0, 0.0])

    expected = F.softplus(negative - positive).mean()
    assert torch.allclose(BPRLoss()(positive, negative), expected)


def test_bpr_contrasts_a_row_with_each_of_its_negatives():
    """Several negatives per row, as the sequential models draw them."""
    positive = torch.tensor([1.0, 2.0])
    negative = torch.tensor([[0.0, 3.0], [1.0, 2.0]])

    expected = F.softplus(negative - positive.unsqueeze(1)).mean()
    assert torch.allclose(BPRLoss()(positive, negative), expected)
