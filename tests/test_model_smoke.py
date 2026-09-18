"""Every registered model must build, train a step, predict and be evaluated.

This suite is deliberately shallow and wide: it makes no claim about the quality
of a model, only that the whole path from configuration to a score matrix holds
together for all of them. Most of the defects this framework has shipped were of
exactly that shape, and none of them needed a deep test to catch.
"""

from typing import Any, Dict

import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import (
    ContextRecommenderUtils,
    IterativeRecommender,
    SequentialRecommenderUtils,
)
from warprec.utils.registry import model_registry, params_registry

from conftest import build_params

MODELS = sorted(
    name
    for name in model_registry.list_registered()
    # ProxyRecommender replays a file produced elsewhere; it has no hyperparameters
    # and is covered by the evaluation tests instead.
    if name != "PROXYRECOMMENDER"
)


@pytest.mark.parametrize("model_name", MODELS)
def test_model_builds_trains_and_predicts(model_name: str, dataset: Dataset):
    """A model must construct, take a training step and score every item."""
    params = build_params(model_name)

    # The schema is the contract the configuration file is validated against, so
    # a parameter set that fails here would fail for a user too.
    params_registry.get(model_name, **params)

    model = model_registry.get(
        model_name,
        params=params,
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
    )

    if isinstance(model, IterativeRecommender):
        loader = model.get_dataloader(
            interactions=dataset.train_set, sessions=dataset.train_session
        )
        batch = next(iter(loader))

        # Some models set per-epoch state (KL annealing, for instance) in the
        # Lightning hook that precedes the first step, so the loop is emulated
        # rather than short-circuited.
        model.on_train_epoch_start()

        loss = model.training_step(batch, 0)
        assert torch.isfinite(loss).all(), f"{model_name}: non-finite training loss"

    model.eval()
    users = torch.arange(min(4, dataset.info()["n_users"]))
    predict_kwargs: Dict[str, Any] = {"user_indices": users}
    if isinstance(model, SequentialRecommenderUtils):
        history, lengths, _ = dataset.train_set.get_history()
        predict_kwargs["user_seq"] = history[users][:, -model.max_seq_len :]
        predict_kwargs["seq_len"] = lengths[users].clamp(max=model.max_seq_len)
    if isinstance(model, ContextRecommenderUtils) and model.context_dims:
        _, _, _, contexts = dataset.train_transactions.get_arrays()
        predict_kwargs["contexts"] = torch.from_numpy(contexts[: len(users)])

    with torch.inference_mode():
        scores = model.predict(**predict_kwargs)

    assert scores.shape[0] == len(users), f"{model_name}: wrong number of rows"
    assert scores.shape[1] == dataset.info()["n_items"], f"{model_name}: wrong width"
    assert torch.isfinite(scores).any(), f"{model_name}: no finite score produced"
