"""Context-aware models are served with the situation each request describes.

A context arrives as raw values - 'morning', 21.5, ['jazz', 'live'] - and has
to reach the model encoded exactly as training encoded it. The encoding is
checked against the dataset's own encoding of the same training rows, so
serving cannot drift from training without failing here.
"""

import io
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import torch

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import ContextRecommenderUtils
from warprec.serving.payload import build_serving_payload
from warprec.serving.servable import ServableModel, ServingError, ServingPolicy
from warprec.utils.registry import model_registry

from conftest import CONTEXT_LABELS, make_model, save_servable

CONTEXT_MODELS = sorted(
    name
    for name in model_registry.list_registered()
    if issubclass(model_registry.get_class(name), ContextRecommenderUtils)
)
RICH_LABELS = ["daytime", "temperature", "tags"]


@pytest.fixture(scope="module")
def rich_frame() -> pd.DataFrame:
    """Interactions whose context uses every encoding: a category, a number and a
    multi-valued field. There are no item features, the other shape a context
    model's feature lookup can take."""
    rng = np.random.default_rng(7)
    pairs = [
        (u, int(i)) for u in range(30) for i in rng.choice(20, size=6, replace=False)
    ]
    return pd.DataFrame(
        {
            "user_id": [u for u, _ in pairs],
            "item_id": [i for _, i in pairs],
            "rating": rng.integers(1, 6, len(pairs)).astype(float),
            "timestamp": rng.integers(1_000_000, 2_000_000, len(pairs)),
            "daytime": rng.choice(["morning", "evening", "night"], len(pairs)),
            "temperature": rng.normal(20, 5, len(pairs)).round(1),
            "tags": [
                "|".join(
                    rng.choice(
                        ["jazz", "live", "solo", "duo"],
                        size=rng.integers(1, 4),
                        replace=False,
                    )
                )
                for _ in pairs
            ],
        }
    )


@pytest.fixture(scope="module")
def rich_dataset(rich_frame: pd.DataFrame) -> Dataset:
    train = rich_frame.groupby("user_id", group_keys=False).apply(lambda g: g.iloc[:-1])
    evaluation = rich_frame.groupby("user_id", group_keys=False).apply(
        lambda g: g.iloc[-1:]
    )
    return Dataset(
        train_data=train,
        eval_data=evaluation,
        rating_type="explicit",
        rating_label="rating",
        timestamp_label="timestamp",
        context_labels=RICH_LABELS,
        context_separators={"tags": "|"},
        batch_size=64,
    )


def through_disk(state: Dict[str, Any]) -> Dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def served(tmp_path: Path, data: Dataset, name: str = "FM", **policy) -> ServableModel:
    path = save_servable(tmp_path / f"{name}.pth", make_model(name, data), data)
    return ServableModel.from_checkpoint(path, policy=ServingPolicy(**policy))


def test_context_models_are_discovered():
    assert "FM" in CONTEXT_MODELS and len(CONTEXT_MODELS) >= 8


@pytest.mark.parametrize("which", ["dataset", "rich_dataset"])
@pytest.mark.parametrize("model_name", CONTEXT_MODELS)
def test_context_models_restore_from_the_checkpoint(
    model_name: str, which: str, request
):
    """With item features and without, the restored model scores identically."""
    data = request.getfixturevalue(which)
    model = make_model(model_name, data)
    model.eval()
    contexts = data.train_transactions.get_arrays()[3]
    inputs = {
        "user_indices": torch.tensor([0, 1]),
        "contexts": torch.from_numpy(contexts[:2]),
    }
    with torch.inference_mode():
        before = model.predict(**inputs)

    restored = model_registry.get_class(model_name).from_checkpoint(
        checkpoint=through_disk(model.get_state())
    )
    restored.eval()
    with torch.inference_mode():
        after = restored.predict(**inputs)
    assert torch.equal(before, after)


def test_the_payload_carries_the_context_vocabulary(dataset: Dataset):
    assert (
        build_serving_payload(make_model("FM", dataset), dataset)["context_maps"]
        == dataset.get_context_maps()
    )
    assert (
        build_serving_payload(make_model("BPR", dataset), dataset)["context_maps"]
        is None
    )


@pytest.mark.parametrize(
    "which, frame_name, labels",
    [
        ("dataset", "interactions_frame", CONTEXT_LABELS),
        ("rich_dataset", "rich_frame", RICH_LABELS),
    ],
)
def test_a_context_is_encoded_exactly_as_training_encoded_it(
    tmp_path: Path, request, which: str, frame_name: str, labels
):
    data = request.getfixturevalue(which)
    raw = request.getfixturevalue(frame_name).set_index(["user_id", "item_id"])
    servable = served(tmp_path, data)
    users, items, _, contexts = data.train_transactions.get_arrays()
    user_labels, item_labels = data.get_inverse_mappings()

    for row in range(0, len(users), max(1, len(users) // 25)):
        values = raw.loc[(user_labels[users[row]], item_labels[items[row]]), labels]
        context = {
            label: values[label].split("|") if label == "tags" else values[label]
            for label in labels
        }
        query = servable.resolve(user_id=user_labels[users[row]], context=context)
        encoded = servable.context_schema.tensor([query.context], "cpu")[0].numpy()
        assert np.array_equal(encoded, contexts[row]), f"row {row}: {context}"


def test_recommendations_follow_the_context(tmp_path: Path, dataset: Dataset):
    model = make_model("FM", dataset)
    model.eval()
    servable = served(tmp_path, dataset)
    user_labels, item_labels = dataset.get_inverse_mappings()
    maps = dataset.get_context_maps()

    for context in (
        {"daytime": "morning", "weather": "sunny"},
        {"daytime": "evening", "weather": "rainy"},
    ):
        (answer,) = servable.recommend(
            [servable.resolve(user_id=user_labels[2], k=5, context=context)]
        )
        encoded = torch.tensor(
            [[float(maps[label][context[label]]) for label in model.context_labels]]
        )
        with torch.inference_mode():
            scores = model.predict(user_indices=torch.tensor([2]), contexts=encoded)[0]
        # Cloned outside inference mode, so that it can be edited in place.
        scores = scores.clone()
        scores[dataset.train_set.get_sparse()[2].indices] = -math.inf
        expected = [item_labels[i] for i in torch.topk(scores, 5).indices.tolist()]
        assert [entry["item_id"] for entry in answer] == expected


def test_the_context_changes_the_scores(tmp_path: Path, dataset: Dataset):
    servable = served(tmp_path, dataset)
    users, items = dataset.get_inverse_mappings()
    candidates = [items[0], items[1], items[2]]
    morning = servable.score(
        items=candidates,
        user_id=users[0],
        context={"daytime": "morning", "weather": "sunny"},
    )
    evening = servable.score(
        items=candidates,
        user_id=users[0],
        context={"daytime": "evening", "weather": "rainy"},
    )
    assert [e["score"] for e in morning] != [e["score"] for e in evening]


def test_a_context_may_be_a_list_in_field_order(tmp_path: Path, rich_dataset: Dataset):
    servable = served(tmp_path, rich_dataset)
    users, _ = rich_dataset.get_inverse_mappings()
    order = list(servable.describe()["context"])
    as_dict = {"daytime": "night", "temperature": 18.5, "tags": ["jazz", "live"]}
    as_list = [as_dict[label] for label in order]
    by_dict = servable.recommend([servable.resolve(user_id=users[1], context=as_dict)])
    by_list = servable.recommend([servable.resolve(user_id=users[1], context=as_list)])
    assert by_dict == by_list


@pytest.mark.parametrize(
    "context, message",
    [
        (None, "context-aware"),
        ({"daytime": "morning"}, "missing"),
        ({"daytime": "morning", "weather": "sunny", "mood": "happy"}, "unexpected"),
        ({"daytime": "brunch", "weather": "sunny"}, "known value"),
        ({"daytime": ["morning", "evening"], "weather": "sunny"}, "one value"),
        (["morning"], "one value per field"),
    ],
)
def test_a_bad_context_is_refused(
    tmp_path: Path, dataset: Dataset, context, message: str
):
    servable = served(tmp_path, dataset)
    users, _ = dataset.get_inverse_mappings()
    with pytest.raises(ServingError, match=message) as error:
        servable.resolve(user_id=users[0], context=context)
    assert error.value.status == 422


@pytest.mark.parametrize(
    "context, message",
    [
        ({"daytime": "night", "temperature": "warm", "tags": ["jazz"]}, "numeric"),
        (
            {"daytime": "night", "temperature": 20, "tags": ["jazz", "polka"]},
            "known value",
        ),
    ],
)
def test_numbers_and_multi_valued_fields_are_checked(
    tmp_path: Path, rich_dataset: Dataset, context, message: str
):
    servable = served(tmp_path, rich_dataset)
    users, _ = rich_dataset.get_inverse_mappings()
    with pytest.raises(ServingError, match=message):
        servable.resolve(user_id=users[0], context=context)


def test_a_model_without_context_refuses_one(tmp_path: Path, dataset: Dataset):
    users, _ = dataset.get_inverse_mappings()
    with pytest.raises(ServingError, match="does not use context"):
        served(tmp_path, dataset, "BPR").resolve(
            user_id=users[0], context={"daytime": "morning"}
        )


def test_a_batch_of_different_contexts_answers_each_as_if_alone(
    tmp_path: Path, rich_dataset: Dataset
):
    servable = served(tmp_path, rich_dataset, unknown_user="popular")
    users, _ = rich_dataset.get_inverse_mappings()
    contexts = [
        {"daytime": "morning", "temperature": 12.0, "tags": ["solo"]},
        {"daytime": "night", "temperature": 25.5, "tags": ["jazz", "live", "duo"]},
        {"daytime": "evening", "temperature": 19.0, "tags": ["live"]},
    ]
    queries = [
        servable.resolve(user_id=users[u], k=2 + u, context=contexts[u])
        for u in range(3)
    ]
    queries.insert(1, servable.resolve(user_id="nobody", k=3, context=contexts[0]))
    assert servable.recommend(queries) == [servable.recommend([q])[0] for q in queries]


def test_describe_lists_the_known_context_values(tmp_path: Path, rich_dataset: Dataset):
    context = served(tmp_path, rich_dataset).describe()["context"]
    assert context["daytime"] == {
        "type": "token",
        "values": ["evening", "morning", "night"],
    }
    assert context["temperature"] == {"type": "float", "values": None}
    assert context["tags"]["type"] == "seq"
    assert set(context["tags"]["values"]) == {"jazz", "live", "solo", "duo"}


def test_an_old_context_checkpoint_is_refused_with_a_reason(
    tmp_path: Path, dataset: Dataset
):
    path = tmp_path / "old.pth"
    torch.save(make_model("FM", dataset).get_state(), path)
    with pytest.raises(ValueError, match="context values"):
        ServableModel.from_checkpoint(path)


def test_an_empty_multi_valued_field_is_padding_as_in_training():
    """Training encodes an empty cell as the padding index; serving must too,
    or the request breaks the whole batch it is scored with."""
    from warprec.serving.context import ContextSchema

    schema = ContextSchema(
        labels=["tags"], types={"tags": "seq"}, maps={"tags": {"jazz": 1}}, max_len=1
    )
    encoded = schema.tensor(
        [schema.encode({"tags": []}), schema.encode({"tags": ["jazz"]})], "cpu"
    )
    assert encoded.tolist() == [[0.0], [1.0]]


def test_an_empty_multi_valued_field_does_not_break_its_batch(
    tmp_path: Path, rich_dataset: Dataset
):
    servable = served(tmp_path, rich_dataset)
    users, _ = rich_dataset.get_inverse_mappings()
    queries = [
        servable.resolve(
            user_id=users[0],
            k=3,
            context={"daytime": "night", "temperature": 20, "tags": []},
        ),
        servable.resolve(
            user_id=users[1],
            k=3,
            context={"daytime": "morning", "temperature": 10, "tags": ["jazz"]},
        ),
    ]
    assert [len(answer) for answer in servable.recommend(queries)] == [3, 3]


def test_numeric_and_multi_valued_context_stats(
    rich_dataset: Dataset, rich_frame: pd.DataFrame
):
    """A number is summarised by its range; a multi-valued field counts each value."""
    stats = build_serving_payload(make_model("FM", rich_dataset), rich_dataset)[
        "context_stats"
    ]
    train = rich_frame.groupby("user_id", group_keys=False).apply(lambda g: g.iloc[:-1])
    assert stats["temperature"]["type"] == "float"
    assert stats["temperature"]["min"] == pytest.approx(
        train["temperature"].min(), abs=1e-4
    )
    assert stats["temperature"]["max"] == pytest.approx(
        train["temperature"].max(), abs=1e-4
    )
    tags = train["tags"].str.split("|").explode().value_counts().to_dict()
    assert stats["tags"] == {"type": "seq", "counts": tags}
