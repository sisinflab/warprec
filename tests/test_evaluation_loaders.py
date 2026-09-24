"""The datasets the evaluator iterates, for the contextual protocols.

A contextual evaluation asks a different question of every row — this user, this
item, in this situation — so the loader has to keep the three aligned. If the
context drifted out of step with the pair beside it, every contextual number the
framework reports would be measuring the wrong situation, and nothing about the
output would look wrong.
"""

from typing import List

import narwhals as nw
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from warprec.data.eval_loaders import (
    ContextualEvaluationDataset,
    SampledContextualEvaluationDataset,
)

CONTEXTS: List[str] = ["daytime", "weather"]


def frame() -> nw.DataFrame:
    """Four evaluation rows, each in a distinct situation.

    Returns:
        nw.DataFrame: The evaluation rows.
    """
    return nw.from_native(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 2],
                "item_id": [3, 4, 3, 1],
                "daytime": [1.0, 2.0, 1.0, 2.0],
                "weather": [1.0, 1.0, 2.0, 2.0],
            }
        ),
        eager_only=True,
    )


def test_every_row_keeps_its_own_situation():
    """The context has to stay with the pair it was recorded against."""
    dataset = ContextualEvaluationDataset(frame(), "user_id", "item_id", CONTEXTS)

    assert len(dataset) == 4

    for position, (user, item, context) in enumerate(
        [(0, 3, [1.0, 1.0]), (0, 4, [2.0, 1.0]), (1, 3, [1.0, 2.0]), (2, 1, [2.0, 2.0])]
    ):
        got_user, got_item, got_context = dataset[position]
        assert int(got_user) == user
        assert int(got_item) == item
        assert torch.allclose(
            torch.as_tensor(got_context, dtype=torch.float),
            torch.tensor(context),
        )


def test_the_same_pair_in_two_situations_is_two_rows():
    """Repeated pairs are the signal a contextual evaluation exists to read."""
    dataset = ContextualEvaluationDataset(frame(), "user_id", "item_id", CONTEXTS)

    first, second = dataset[0], dataset[1]

    assert int(first[0]) == int(second[0])
    # Same user, different item and different situation: both must survive.
    assert not torch.allclose(
        torch.as_tensor(first[2], dtype=torch.float),
        torch.as_tensor(second[2], dtype=torch.float),
    )


def test_the_contexts_are_read_as_numbers_the_models_can_consume():
    """Categorical fields arrive already encoded, so the tensor stays numeric."""
    dataset = ContextualEvaluationDataset(frame(), "user_id", "item_id", CONTEXTS)

    _, _, context = dataset[0]
    tensor = torch.as_tensor(context)

    assert tensor.dtype in (torch.float32, torch.float64)
    assert tensor.numel() == len(CONTEXTS)
    assert bool(np.isfinite(tensor.numpy()).all())


def sampled_contextual(
    num_negatives: int, num_items: int = 12
) -> SampledContextualEvaluationDataset:
    """Build the sampled contextual dataset over a small catalogue.

    Args:
        num_negatives (int): How many negatives to ask for per row.
        num_items (int): The catalogue size.

    Returns:
        SampledContextualEvaluationDataset: The dataset under test.
    """
    # Each user has seen two items in training, so the rest are drawable.
    train = csr_matrix(
        (
            np.ones(6),
            ([0, 0, 1, 1, 2, 2], [0, 1, 2, 3, 4, 5]),
        ),
        shape=(3, num_items),
    )
    return SampledContextualEvaluationDataset(
        train_interactions=train,
        eval_data=frame(),
        user_id_label="user_id",
        item_id_label="item_id",
        context_labels=CONTEXTS,
        num_items=num_items,
        num_negatives=num_negatives,
        seed=11,
    )


def test_the_candidates_exclude_what_the_user_already_saw():
    """A negative the user interacted with in training is not a negative."""
    dataset = sampled_contextual(num_negatives=4)

    for position in range(len(dataset)):
        _, positive, negatives, _ = dataset[position]
        drawn = set(int(item) for item in torch.as_tensor(negatives).flatten())

        assert len(drawn) == 4, "the candidate list was not the width asked for"
        assert int(positive) not in drawn, "the answer was offered as a negative"


def test_every_row_draws_the_same_number_of_candidates():
    """The evaluator lines the batch up, so the rows have to agree on width."""
    dataset = sampled_contextual(num_negatives=3)

    widths = {
        torch.as_tensor(dataset[position][2]).numel()
        for position in range(len(dataset))
    }

    assert widths == {3}


def test_asking_for_more_candidates_than_the_catalogue_holds_is_refused():
    """It used to draw for them forever rather than saying so.

    The top-up loop rejected anything the user had seen, with no bound, so a
    catalogue too small to satisfy the request meant a run that hung silently.
    """
    with pytest.raises(ValueError, match="left unseen"):
        sampled_contextual(num_negatives=20, num_items=12)
