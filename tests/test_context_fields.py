"""Context-aware models train and evaluate with every kind of context field.

A context field is a category, a number or a set of values. Training and
evaluation read the same encoded columns, so a field that only one of them
understands fails late - at the first evaluation, or at the first numeric value
above one - after the run has already spent its time.
"""

import math
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.evaluation.evaluator import Evaluator
from warprec.recommenders.base_recommender import ContextRecommenderUtils
from warprec.utils.registry import model_registry

from conftest import make_model

CONTEXT_MODELS = sorted(
    name
    for name in model_registry.list_registered()
    if issubclass(model_registry.get_class(name), ContextRecommenderUtils)
)


def context_frame(multi_valued: bool) -> pd.DataFrame:
    """Interactions with a category, a number and, optionally, a set of values.

    The numbers go well above one and below zero: an index taken from them
    would point past the field's single row, or into another field's.

    Args:
        multi_valued (bool): Whether to add the multi-valued field.

    Returns:
        pd.DataFrame: The interactions.
    """
    rng = np.random.default_rng(3)
    pairs = [
        (u, int(i)) for u in range(30) for i in rng.choice(20, size=6, replace=False)
    ]
    frame = pd.DataFrame(
        {
            "user_id": [u for u, _ in pairs],
            "item_id": [i for _, i in pairs],
            "rating": rng.integers(1, 6, len(pairs)).astype(float),
            "timestamp": rng.integers(1_000_000, 2_000_000, len(pairs)),
            "daytime": rng.choice(["morning", "evening", "night"], len(pairs)),
            "temperature": rng.uniform(-10, 35, len(pairs)).round(1),
        }
    )
    if multi_valued:
        frame["tags"] = [
            "|".join(
                rng.choice(
                    ["jazz", "live", "solo", "duo"],
                    size=rng.integers(1, 4),
                    replace=False,
                )
            )
            for _ in pairs
        ]
    return frame


def context_dataset(multi_valued: bool) -> Dataset:
    """A dataset holding out each user's last interaction, with its contexts.

    Args:
        multi_valued (bool): Whether to add the multi-valued field.

    Returns:
        Dataset: The dataset.
    """
    frame = context_frame(multi_valued)
    labels: List[str] = ["daytime", "temperature"] + (["tags"] if multi_valued else [])
    return Dataset(
        train_data=frame.groupby("user_id", group_keys=False).apply(
            lambda g: g.iloc[:-1]
        ),
        eval_data=frame.groupby("user_id", group_keys=False).apply(
            lambda g: g.iloc[-1:]
        ),
        rating_type="explicit",
        rating_label="rating",
        timestamp_label="timestamp",
        context_labels=labels,
        context_separators={"tags": "|"} if multi_valued else {},
        batch_size=16,
    )


@pytest.fixture(scope="module")
def numeric_dataset() -> Dataset:
    return context_dataset(multi_valued=False)


@pytest.fixture(scope="module")
def multi_valued_dataset() -> Dataset:
    return context_dataset(multi_valued=True)


def test_every_context_model_is_covered():
    expected = ["AFM", "DCN", "DCNv2", "DeepFM", "FM", "NFM", "WideAndDeep", "xDeepFM"]
    assert CONTEXT_MODELS == sorted(name.upper() for name in expected)


@pytest.mark.parametrize("strategy", ["full", "sampled"])
def test_the_evaluation_contexts_are_encoded_as_training_encodes_them(
    multi_valued_dataset: Dataset, strategy: str
):
    """A multi-valued cell such as '1 3' used to be cast to a float and fail."""
    if strategy == "full":
        loader = multi_valued_dataset.get_contextual_evaluation_dataloader()
    else:
        loader = multi_valued_dataset.get_sampled_contextual_evaluation_dataloader(
            num_negatives=5
        )
    contexts = torch.cat([batch[-1] for batch in loader])
    training = multi_valued_dataset.train_transactions.get_arrays()[3]

    assert contexts.shape[1:] == training.shape[1:]
    tags = multi_valued_dataset.train_transactions.context_labels.index("tags")
    # Every row names at least one tag, and none beyond the vocabulary.
    assert (contexts[:, tags, 0] >= 1).all()
    assert contexts[:, tags].max() < multi_valued_dataset.info()["context_dims"]["tags"]


def assert_evaluates(
    model: ContextRecommenderUtils, data: Dataset, strategy: str
) -> None:
    """Evaluate a model on the contextual loader and check every result is a number.

    Args:
        model (ContextRecommenderUtils): The context-aware model.
        data (Dataset): The dataset it was built on.
        strategy (str): 'full' or 'sampled'.
    """
    evaluator = Evaluator(
        ["nDCG", "HitRate"], [5], train_set=data.train_set.get_sparse()
    )
    if strategy == "full":
        loader = data.get_contextual_evaluation_dataloader()
    else:
        loader = data.get_sampled_contextual_evaluation_dataloader(num_negatives=5)

    evaluator.evaluate(model, loader, strategy, data)

    results = evaluator.compute_results()[5]
    for name in ("nDCG", "HitRate"):
        value = results[name]
        value = value.float().nanmean().item() if torch.is_tensor(value) else value
        assert math.isfinite(value), f"{model.name} {strategy}: {name} is {value}"


@pytest.mark.parametrize("strategy", ["full", "sampled"])
@pytest.mark.parametrize("model_name", CONTEXT_MODELS)
def test_a_multi_valued_field_is_evaluated(
    multi_valued_dataset: Dataset, model_name: str, strategy: str
):
    assert_evaluates(
        make_model(model_name, multi_valued_dataset), multi_valued_dataset, strategy
    )


@pytest.mark.parametrize("which", ["numeric_dataset", "multi_valued_dataset"])
@pytest.mark.parametrize("model_name", CONTEXT_MODELS)
def test_numeric_and_multi_valued_fields_train_and_evaluate(
    model_name: str, which: str, request
):
    """A numeric value of one or more used to be taken as an index by the
    regularisation, past the single row its field owns: IndexError."""
    data = request.getfixturevalue(which)
    model = make_model(model_name, data)
    loader = model.get_dataloader(
        interactions=data.train_set, sessions=data.train_session
    )
    model.on_train_epoch_start()
    for step, batch in enumerate(loader):
        loss = model.training_step(batch, step)
        assert torch.isfinite(loss).all(), f"{model_name}: non-finite loss"
    for strategy in ("full", "sampled"):
        assert_evaluates(model, data, strategy)


def pooled(rows: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "sum":
        return rows.sum(dim=0)
    if pooling == "max":
        return rows.max(dim=0).values
    return rows.mean(dim=0)


@pytest.mark.parametrize("pooling", ["mean", "sum", "max"])
def test_every_value_of_a_field_reaches_the_bias_and_the_regularisation(
    multi_valued_dataset: Dataset, pooling: str
):
    """The bias of a multi-valued field pools all its values, as the embedding
    does, and a numeric field always reads its own row, scaled by its value."""
    torch.manual_seed(0)
    model = make_model("FM", multi_valued_dataset)
    model.context_pooling = pooling
    labels = model.context_labels
    daytime, temperature, tags = (
        labels.index("daytime"),
        labels.index("temperature"),
        labels.index("tags"),
    )
    offsets = model.context_offsets.tolist()
    width = multi_valued_dataset.train_transactions.get_arrays()[3].shape[2]

    contexts = torch.zeros(1, len(labels), width)
    contexts[0, daytime, 0] = 2
    contexts[0, temperature, 0] = 21.5
    contexts[0, tags, :2] = torch.tensor([1.0, 3.0])

    bias = model.merged_context_bias.weight.detach().squeeze(-1)
    table = model.merged_context_embedding.weight.detach()
    tag_rows = [offsets[tags] + 1, offsets[tags] + 3]

    expected_bias = (
        bias[offsets[daytime] + 2]
        + 21.5 * bias[offsets[temperature]]
        + pooled(bias[tag_rows], pooling)
    )
    with torch.no_grad():
        assert torch.allclose(model._get_context_bias(contexts)[0], expected_bias)

        embeddings = model._get_context_embeddings(contexts)[0]
        assert torch.allclose(
            embeddings[temperature], 21.5 * table[offsets[temperature]]
        )
        assert torch.allclose(embeddings[tags], pooled(table[tag_rows], pooling))

        regularised = model.get_reg_params(
            torch.tensor([0]), torch.tensor([0]), None, contexts
        )[-2:]
    used = [offsets[daytime] + 2, offsets[temperature]] + tag_rows
    assert torch.allclose(regularised[0].pow(2).sum(), table[used].pow(2).sum())
    assert torch.allclose(regularised[1].pow(2).sum(), bias[used].pow(2).sum())
