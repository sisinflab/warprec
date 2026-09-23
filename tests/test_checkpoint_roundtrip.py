"""A saved model must come back able to score, with no training data at hand.

Serving loads a checkpoint and nothing else. Models fitted in a single
closed-form step keep their result in plain attributes rather than in
parameters, so this is the path that decides whether they can be served at all.
"""

from typing import Any, Dict

import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.utils.registry import model_registry

from conftest import build_params

# Models fitted on the interactions: their constructor cannot run without them,
# which is exactly what makes the checkpoint the only thing serving can rely on.
# Closed-form models keep what they learned in plain attributes, so the
# checkpoint is the only thing that can carry it to a serving process.
CLOSED_FORM = sorted(
    name
    for name in model_registry.list_registered()
    if name != "PROXYRECOMMENDER"
    and model_registry.get_class(name)._fits_on_interactions()
    and not issubclass(model_registry.get_class(name), IterativeRecommender)
)

# Iteratively trained models keep their result in parameters, but their
# constructor still derives its shapes from the interactions.
ITERATIVE_NEEDING_INTERACTIONS = sorted(
    name
    for name in model_registry.list_registered()
    if name != "PROXYRECOMMENDER"
    and model_registry.get_class(name)._fits_on_interactions()
    and issubclass(model_registry.get_class(name), IterativeRecommender)
)


def test_the_discovered_sets_are_not_empty():
    """Guards against the discovery above silently matching nothing."""
    assert CLOSED_FORM, "no closed-form models found - check the discovery"
    assert ITERATIVE_NEEDING_INTERACTIONS, (
        "no iterative models found - check the discovery"
    )


@pytest.mark.parametrize("model_name", CLOSED_FORM)
def test_checkpoint_restores_without_interactions(model_name: str, dataset: Dataset):
    """A restored model scores exactly as the one that was saved."""
    params = build_params(model_name)
    model = model_registry.get(
        model_name,
        params=params,
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
        knowledge=dataset.knowledge,
        multimodal=dataset.multimodal,
    )
    model.eval()

    users = torch.arange(min(4, dataset.info()["n_users"]))
    predict_kwargs: Dict[str, Any] = {"user_indices": users}
    if isinstance(model, SequentialRecommenderUtils):
        history, lengths, _ = dataset.train_set.get_history()
        predict_kwargs["user_seq"] = history[users][:, -model.max_seq_len :]
        predict_kwargs["seq_len"] = lengths[users].clamp(max=model.max_seq_len)

    with torch.inference_mode():
        before = model.predict(**predict_kwargs)

    # This is what the serving layer does: the checkpoint and nothing else.
    checkpoint = model.get_state()
    restored = model_registry.get_class(model_name).from_checkpoint(
        checkpoint=checkpoint
    )
    restored.eval()

    with torch.inference_mode():
        after = restored.predict(**predict_kwargs)

    assert torch.equal(before, after), f"{model_name}: scores changed after restoring"


@pytest.mark.parametrize("model_name", ITERATIVE_NEEDING_INTERACTIONS)
def test_iterative_models_say_what_they_need(model_name: str, dataset: Dataset):
    """Loading one of these without interactions fails with an actionable message."""
    params = build_params(model_name)
    model = model_registry.get(
        model_name,
        params=params,
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
        knowledge=dataset.knowledge,
        multimodal=dataset.multimodal,
    )
    model_class = model_registry.get_class(model_name)

    with pytest.raises(ValueError, match="training interactions"):
        model_class.from_checkpoint(checkpoint=model.get_state())

    # With them supplied it rebuilds as before.
    restored = model_class.from_checkpoint(
        checkpoint=model.get_state(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
        knowledge=dataset.knowledge,
        multimodal=dataset.multimodal,
    )
    assert isinstance(restored, model_class)
