"""A saved model must come back able to score, with no training data at hand.

Serving loads a checkpoint and nothing else. Models fitted in a single
closed-form step keep their result in plain attributes rather than in
parameters, and models whose constructor derives a graph, a matrix or a feature
table from the training data are kept whole, so these are the paths that decide
whether a model can be served at all. Every checkpoint goes through torch.save
and torch.load, because pickling is exactly what can fail.
"""

import io
from typing import Any, Dict

import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import IterativeRecommender
from warprec.utils.registry import model_registry

from conftest import build_params, make_model
from test_model_smoke import MODELS, predict_arguments

# Iteratively trained models keep their result in parameters, but their
# constructor still derives its shapes from the training data, so they are
# saved whole.
SAVED_WHOLE = sorted(
    name
    for name in MODELS
    if model_registry.get_class(name)._fits_on_data()
    and issubclass(model_registry.get_class(name), IterativeRecommender)
)


def through_disk(state: Dict[str, Any]) -> Dict[str, Any]:
    """Save and reload a checkpoint the way serving does, pickling included."""
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def scores_survive_the_round_trip(model_name: str, dataset: Dataset) -> None:
    """A restored model scores exactly as the one that was saved."""
    model = make_model(model_name, dataset)
    model.eval()
    inputs = predict_arguments(model, dataset, torch.arange(4))
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


def test_the_discovered_set_is_not_empty():
    """Guards against the discovery above silently matching nothing."""
    assert SAVED_WHOLE, "no model saved whole found - check the discovery"


@pytest.mark.parametrize("model_name", MODELS)
def test_checkpoint_restores_without_training_data(model_name: str, dataset: Dataset):
    """Every model family comes back from its checkpoint alone."""
    scores_survive_the_round_trip(model_name, dataset)


@pytest.mark.parametrize("model_name", SAVED_WHOLE)
def test_an_older_checkpoint_still_says_what_it_needs(
    model_name: str, dataset: Dataset
):
    """A checkpoint saved before the module was kept fails with a clear message."""
    model = make_model(model_name, dataset)
    state = model.get_state()
    del state["module"]
    model_class = model_registry.get_class(model_name)

    with pytest.raises(ValueError, match="training data"):
        model_class.from_checkpoint(checkpoint=state)

    # With the data supplied it rebuilds as before.
    restored = model_class.from_checkpoint(
        checkpoint=state,
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
        knowledge=dataset.knowledge,
        multimodal=dataset.multimodal,
    )
    assert isinstance(restored, model_class)


def test_lightccf_without_layers_keeps_its_scoring_cache_out_of_the_state(
    dataset: Dataset,
):
    """Its forward returns the embedding weights, which scoring used to register."""
    model_class = model_registry.get_class("LightCCF")
    params = {**build_params("LightCCF"), "n_layers": 0}
    model = model_class(
        params=params, info=dataset.info(), interactions=dataset.train_set
    )
    model.eval()
    with torch.inference_mode():
        model.predict(**predict_arguments(model, dataset, torch.arange(4)))

    state = model.get_state()
    assert not [key for key in state["state_dict"] if key.startswith("_cached_")]


def test_a_checkpoint_carrying_the_scoring_cache_still_loads(dataset: Dataset):
    """Checkpoints written with the cache load without it."""
    model = make_model("LightCCF", dataset)
    state = model.get_state()
    state["state_dict"]["_cached_user_emb"] = model.user_embedding.weight.detach()
    state["state_dict"]["_cached_item_emb"] = model.item_embedding.weight.detach()

    restored = model_registry.get_class("LightCCF").from_checkpoint(
        checkpoint=through_disk(state), interactions=dataset.train_set
    )
    assert torch.equal(restored.user_embedding.weight, model.user_embedding.weight)


def test_checkpoints_record_their_format_and_version(dataset: Dataset):
    """Serving can tell which WarpRec wrote a checkpoint, and in which layout."""
    state = make_model("BPR", dataset).get_state()
    assert state["format_version"] == 2
    assert state["warprec_version"] is None or isinstance(state["warprec_version"], str)
    assert "module" not in state, "a model rebuilt from params needs no module"
