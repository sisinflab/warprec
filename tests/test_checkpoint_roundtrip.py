"""A saved model must come back able to score, with no training data at hand.

Serving loads a checkpoint and nothing else. Models fitted in a single
closed-form step keep their result in plain attributes rather than in
parameters, and models whose constructor derives a graph or a matrix from the
interactions are kept whole, so these are the paths that decide whether a
model can be served at all. Every checkpoint goes through torch.save and
torch.load, because pickling is exactly what can fail.
"""

import io
from typing import Any, Dict

import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.utils.registry import model_registry

from conftest import make_model

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


def through_disk(state: Dict[str, Any]) -> Dict[str, Any]:
    """Save and reload a checkpoint the way serving does, pickling included."""
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def predict_inputs(model: Recommender, dataset: Dataset) -> Dict[str, Any]:
    """The inputs that score the first few users with any model family."""
    users = torch.arange(min(4, dataset.info()["n_users"]))
    inputs: Dict[str, Any] = {"user_indices": users}
    if isinstance(model, SequentialRecommenderUtils):
        history, lengths, _ = dataset.train_set.get_history()
        inputs["user_seq"] = history[users][:, -model.max_seq_len :]
        inputs["seq_len"] = lengths[users].clamp(max=model.max_seq_len)
    return inputs


def scores_survive_the_round_trip(model_name: str, dataset: Dataset) -> None:
    """A restored model scores exactly as the one that was saved."""
    model = make_model(model_name, dataset)
    model.eval()
    inputs = predict_inputs(model, dataset)
    with torch.inference_mode():
        before = model.predict(**inputs)

    # This is what the serving layer does: the checkpoint and nothing else.
    restored = model_registry.get_class(model_name).from_checkpoint(
        checkpoint=through_disk(model.get_state())
    )
    restored.eval()
    with torch.inference_mode():
        after = restored.predict(**inputs)

    assert torch.equal(before, after), f"{model_name}: scores changed after restoring"


def test_the_discovered_sets_are_not_empty():
    """Guards against the discovery above silently matching nothing."""
    assert CLOSED_FORM, "no closed-form models found - check the discovery"
    assert ITERATIVE_NEEDING_INTERACTIONS, (
        "no iterative models found - check the discovery"
    )


@pytest.mark.parametrize("model_name", CLOSED_FORM)
def test_checkpoint_restores_without_interactions(model_name: str, dataset: Dataset):
    """Closed-form models come back from their saved attributes."""
    scores_survive_the_round_trip(model_name, dataset)


@pytest.mark.parametrize("model_name", ITERATIVE_NEEDING_INTERACTIONS)
def test_iterative_models_restore_from_the_saved_module(
    model_name: str, dataset: Dataset
):
    """Graph and autoencoder models come back whole, with no training data."""
    scores_survive_the_round_trip(model_name, dataset)


@pytest.mark.parametrize("model_name", ITERATIVE_NEEDING_INTERACTIONS)
def test_an_older_checkpoint_still_says_what_it_needs(
    model_name: str, dataset: Dataset
):
    """A checkpoint saved before the module was kept fails with a clear message."""
    model = make_model(model_name, dataset)
    state = model.get_state()
    del state["module"]
    model_class = model_registry.get_class(model_name)

    with pytest.raises(ValueError, match="training interactions"):
        model_class.from_checkpoint(checkpoint=state)

    # With them supplied it rebuilds as before.
    restored = model_class.from_checkpoint(
        checkpoint=state,
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
        knowledge=dataset.knowledge,
        multimodal=dataset.multimodal,
    )
    assert isinstance(restored, model_class)


def test_checkpoints_record_their_format_and_version(dataset: Dataset):
    """Serving can tell which WarpRec wrote a checkpoint, and in which layout."""
    state = make_model("BPR", dataset).get_state()
    assert state["format_version"] == 2
    assert state["warprec_version"] is None or isinstance(state["warprec_version"], str)
    assert "module" not in state, "a model rebuilt from params needs no module"
