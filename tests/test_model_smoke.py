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


def build_model(model_name: str, dataset: Dataset) -> Any:
    """Construct one registered model against the shared fixture.

    Args:
        model_name (str): The registered name.
        dataset (Dataset): The dataset under test.

    Returns:
        Any: The constructed model.
    """
    params = build_params(model_name)

    # The schema is the contract the configuration file is validated against, so
    # a parameter set that fails here would fail for a user too.
    params_registry.get(model_name, **params)

    return model_registry.get(
        model_name,
        params=params,
        info=dataset.info(),
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        transactions=dataset.train_transactions,
        knowledge=dataset.knowledge,
        multimodal=dataset.multimodal,
    )


def predict_arguments(
    model: Any, dataset: Dataset, users: torch.Tensor
) -> Dict[str, Any]:
    """The arguments a given model's family needs in order to score.

    Args:
        model (Any): The model about to be asked for scores.
        dataset (Dataset): The dataset under test.
        users (torch.Tensor): The users to score.

    Returns:
        Dict[str, Any]: The keyword arguments for predict.
    """
    arguments: Dict[str, Any] = {"user_indices": users}

    if isinstance(model, SequentialRecommenderUtils):
        history, lengths, _ = dataset.train_set.get_history()
        arguments["user_seq"] = history[users][:, -model.max_seq_len :]
        arguments["seq_len"] = lengths[users].clamp(max=model.max_seq_len)

    if isinstance(model, ContextRecommenderUtils) and model.context_dims:
        _, _, _, contexts = dataset.train_transactions.get_arrays()
        arguments["contexts"] = torch.from_numpy(contexts[: len(users)])

    return arguments


@pytest.mark.parametrize("model_name", MODELS)
def test_model_builds_trains_and_predicts(model_name: str, dataset: Dataset):
    """A model must construct, take a training step and score every item."""
    model = build_model(model_name, dataset)

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

    with torch.inference_mode():
        scores = model.predict(**predict_arguments(model, dataset, users))

    assert scores.shape[0] == len(users), f"{model_name}: wrong number of rows"
    assert scores.shape[1] == dataset.info()["n_items"], f"{model_name}: wrong width"
    assert torch.isfinite(scores).any(), f"{model_name}: no finite score produced"


@pytest.mark.parametrize("model_name", MODELS)
def test_scoring_a_few_items_agrees_with_ranking_them_all(
    model_name: str, dataset: Dataset
):
    """Asking about some items must give what ranking every item would give.

    The sampled path exists so that a run does not have to score the whole
    catalogue, and the sampled evaluation strategy depends on it agreeing with
    the full one. A model whose two paths disagree reports different numbers
    under the two strategies for no reason a user could see.
    """
    model = build_model(model_name, dataset)
    model.eval()

    users = torch.arange(min(4, dataset.info()["n_users"]))
    wanted = torch.arange(3).unsqueeze(0).expand(len(users), -1).contiguous()

    with torch.inference_mode():
        arguments = predict_arguments(model, dataset, users)
        full = model.predict(**arguments)
        again = model.predict(**arguments)
        sampled = model.predict(**arguments, item_indices=wanted)

    # A model that scores stochastically has no fixed ranking for the sampled
    # path to be compared against. The condition is measured rather than listed,
    # so a model that becomes deterministic is checked from then on without
    # anyone remembering to. NaN is not evidence of that: a score comparing
    # unequal to itself says nothing about whether the model is reproducible,
    # so the check looks past it.
    if not torch.equal(torch.nan_to_num(full), torch.nan_to_num(again)):
        pytest.skip(f"{model_name} does not score reproducibly")

    assert sampled.shape == wanted.shape, f"{model_name}: wrong sampled shape"
    torch.testing.assert_close(
        sampled,
        full.gather(1, wanted),
        equal_nan=True,
        msg=f"{model_name}: the sampled path disagrees with the full ranking",
    )
