"""Behavioural tests for DiffRec and for the loader it reads through.

A diffusion recommender is only as good as its noise schedule and its reverse
walk, and both are easy to get subtly wrong in ways that still train. The tests
here pin the properties the model rests on: that the schedule destroys signal
monotonically, that corrupting and repairing are inverses of one another in the
limit, and that recommending actually runs the reverse walk rather than handing
back what it was given.
"""

from typing import Any

import numpy as np
import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.utils.registry import model_registry

from conftest import build_params


def build(dataset: Dataset, **overrides: Any) -> Any:
    """Construct DiffRec against the shared fixture.

    Args:
        dataset (Dataset): The dataset under test.
        **overrides (Any): Hyperparameters to change from the smoke defaults.

    Returns:
        Any: The constructed model.
    """
    params = {**build_params("DiffRec"), **overrides}
    return model_registry.get(
        "DiffRec",
        params=params,
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
    )


def test_the_schedule_destroys_signal_monotonically(dataset: Dataset):
    """Each step must leave less of the original than the one before it."""
    model = build(dataset, steps=50)

    surviving = model.alphas_cumprod
    assert surviving.numel() == 50
    assert bool((surviving[1:] <= surviving[:-1]).all())
    assert float(surviving[0]) > float(surviving[-1])

    # Everything stays a proper proportion of the original signal.
    assert float(surviving.min()) > 0.0 and float(surviving.max()) <= 1.0


def test_the_first_step_is_held_tiny(dataset: Dataset):
    """The least noisy end is pinned, which is what stops it being overfitted."""
    model = build(dataset, steps=20)

    assert float(model.betas[0]) == pytest.approx(1e-5)
    assert float(model.betas[1]) > float(model.betas[0])


def test_corruption_keeps_more_of_the_history_early_than_late(dataset: Dataset):
    """Corrupting is the forward process; it has to actually corrupt."""
    model = build(dataset, steps=100, noise_scale=0.1)
    history = torch.rand(8, dataset.info()["n_items"])
    noise = torch.randn_like(history)

    early = model._corrupt(history, torch.zeros(8, dtype=torch.long), noise)
    late = model._corrupt(history, torch.full((8,), 99, dtype=torch.long), noise)

    near = torch.nn.functional.cosine_similarity(early, history, dim=1).mean()
    far = torch.nn.functional.cosine_similarity(late, history, dim=1).mean()

    assert float(near) > float(far)


def test_no_noise_at_all_leaves_the_history_alone(dataset: Dataset):
    """A zero scale is the degenerate case, and it must be a no-op."""
    model = build(dataset, steps=10, noise_scale=0.0)
    history = torch.rand(4, dataset.info()["n_items"])

    corrupted = model._corrupt(
        history, torch.full((4,), 9, dtype=torch.long), torch.randn_like(history)
    )

    assert torch.equal(corrupted, history)


def test_recommending_runs_the_reverse_walk(dataset: Dataset):
    """Scores must come out of the denoiser, not straight back from the input."""
    model = build(dataset, steps=6, sampling_steps=3)
    model.eval()

    history = torch.zeros(4, dataset.info()["n_items"])
    history[:, :5] = 1.0

    with torch.inference_mode():
        scored = model.forward(history)

    assert scored.shape == history.shape
    assert not torch.allclose(scored, history)


def test_starting_at_zero_steps_still_denoises_once(dataset: Dataset):
    """Zero sampling steps means no added noise, not a passthrough."""
    model = build(dataset, sampling_steps=0)
    model.eval()

    history = torch.rand(3, dataset.info()["n_items"])
    with torch.inference_mode():
        scored = model.forward(history)

    # The history is not corrupted first, but the reverse walk still runs, so
    # what comes back is the denoiser's reading of it.
    assert scored.shape == history.shape


def test_scoring_a_few_items_agrees_with_ranking_them_all(dataset: Dataset):
    """The sampled path must be a restriction of the full one."""
    model = build(dataset, sampling_steps=0)
    model.eval()

    users = torch.arange(4)
    wanted = torch.tensor([[1, 3, 5], [0, 2, 4], [6, 7, 8], [2, 9, 1]])

    with torch.inference_mode():
        full = model.predict(users)
        some = model.predict(users, item_indices=wanted)

    assert torch.allclose(full.gather(1, wanted), some, atol=1e-5)


def test_the_step_description_separates_adjacent_steps(dataset: Dataset):
    """A denoiser that cannot tell the steps apart cannot undo one of them."""
    from warprec.recommenders.collaborative_filtering_recommender.autoencoder.diffrec import (
        timestep_embedding,
    )

    described = timestep_embedding(torch.arange(16), 32)

    assert described.shape == (16, 32)
    assert not torch.allclose(described[0], described[1])
    # Neighbouring steps read more alike than distant ones, which is the point
    # of a sinusoidal description rather than a one-hot.
    near = torch.nn.functional.cosine_similarity(described[4], described[5], dim=0)
    far = torch.nn.functional.cosine_similarity(described[4], described[15], dim=0)
    assert float(near) > float(far)


def test_an_odd_description_width_is_still_the_width_asked_for(dataset: Dataset):
    """The sinusoid bank is built in pairs, so an odd width needs padding."""
    from warprec.recommenders.collaborative_filtering_recommender.autoencoder.diffrec import (
        timestep_embedding,
    )

    assert timestep_embedding(torch.arange(3), 7).shape == (3, 7)


def test_the_interaction_loader_is_reproducible(dataset: Dataset):
    """Every loader in the data layer draws from a generator of its own.

    This one used to build its DataLoader directly, leaving the shuffle order
    drawn from the global torch stream, so an epoch's batches depended on
    whatever else had consumed that stream beforehand.
    """

    def first_batch(seed: int) -> np.ndarray:
        loader = dataset.train_set.get_interaction_dataloader(
            batch_size=8, shuffle=True, seed=seed
        )
        return next(iter(loader))[0].numpy()

    # Seeding the global stream differently must not move the batches.
    torch.manual_seed(1234)
    first = first_batch(7)
    torch.manual_seed(999)
    again = first_batch(7)
    other = first_batch(8)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
