"""A served model answers exactly what the same model would compute offline.

Every test here compares the serving path against a direct call to the model,
so that ID mapping, masking, padding and batching can only pass by agreeing
with the model itself.
"""

import math
from pathlib import Path
from typing import List

import pytest
import torch

from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import Recommender
from warprec.serving.servable import ServableModel, ServingError, ServingPolicy

from warprec.serving.catalogue import read_item_names
from warprec.utils.config.serving_configuration import ItemMetadata

from conftest import make_model, save_servable


def served(tmp_path: Path, name: str, dataset: Dataset, **policy) -> ServableModel:
    path = save_servable(tmp_path / f"{name}.pth", make_model(name, dataset), dataset)
    return ServableModel.from_checkpoint(path, policy=ServingPolicy(**policy))


def labels(dataset: Dataset):
    users, items = dataset.get_inverse_mappings()
    return users, items


def ids(answers: List[List[dict]]) -> List[List]:
    return [[entry["item_id"] for entry in answer] for answer in answers]


def scores(answers: List[List[dict]]) -> List[float]:
    return [entry["score"] for answer in answers for entry in answer]


def reference(
    model: Recommender, dataset: Dataset, user: int, k: int, mask: bool = True
) -> List:
    """The top-k external item ids computed straight from the model."""
    model.eval()
    inputs = {"user_indices": torch.tensor([user])}
    if hasattr(model, "max_seq_len"):
        seq, lens = dataset.train_session.get_user_history_sequences(
            [user], model.max_seq_len
        )
        inputs.update(user_seq=seq, seq_len=lens)
    with torch.inference_mode():
        scores = model.predict(**inputs)[0]
    # Cloned outside inference mode, so that it can be edited in place.
    scores = scores.clone()
    if mask:
        scores[dataset.train_set.get_sparse()[user].indices] = -math.inf
    top = torch.topk(scores, k)
    _, items = labels(dataset)
    return [
        items[i]
        for i, s in zip(top.indices.tolist(), top.values.tolist())
        if math.isfinite(s)
    ]


@pytest.mark.parametrize("name", ["BPR", "LightGCN", "EASE", "SASRec"])
def test_recommendations_match_the_model(tmp_path: Path, dataset: Dataset, name: str):
    servable = served(tmp_path, name, dataset)
    model = make_model(name, dataset)
    users, _ = labels(dataset)
    for user in range(5):
        (answer,) = servable.recommend([servable.resolve(user_id=users[user], k=5)])
        assert [entry["item_id"] for entry in answer] == reference(
            model, dataset, user, 5
        )


def test_ids_match_whether_sent_as_int_or_string(tmp_path: Path, dataset: Dataset):
    servable = served(tmp_path, "BPR", dataset)
    users, _ = labels(dataset)
    as_int = servable.resolve(user_id=int(users[3]))
    as_str = servable.resolve(user_id=str(users[3]))
    assert as_int.user == as_str.user == 3


def test_seen_items_are_left_out(tmp_path: Path, dataset: Dataset):
    servable = served(tmp_path, "BPR", dataset)
    users, items = labels(dataset)
    seen = {items[i] for i in dataset.train_set.get_sparse()[0].indices}
    (answer,) = servable.recommend(
        [servable.resolve(user_id=users[0], k=dataset.info()["n_items"])]
    )
    assert not seen & {entry["item_id"] for entry in answer}


def test_masking_can_be_turned_off(tmp_path: Path, dataset: Dataset):
    servable = served(tmp_path, "BPR", dataset, mask_seen=False)
    users, _ = labels(dataset)
    (answer,) = servable.recommend([servable.resolve(user_id=users[0], k=5)])
    assert [e["item_id"] for e in answer] == reference(
        make_model("BPR", dataset), dataset, 0, 5, mask=False
    )


def test_masking_never_pads_the_list_with_hidden_items(
    tmp_path: Path, dataset: Dataset
):
    servable = served(tmp_path, "BPR", dataset, max_k=1000)
    users, _ = labels(dataset)
    n_items = dataset.info()["n_items"]
    n_seen = dataset.train_set.get_sparse()[0].nnz
    (answer,) = servable.recommend([servable.resolve(user_id=users[0], k=n_items)])
    assert len(answer) == n_items - n_seen
    assert all(math.isfinite(entry["score"]) for entry in answer)


def test_excluded_items_are_left_out(tmp_path: Path, dataset: Dataset):
    servable = served(tmp_path, "BPR", dataset, mask_seen=False)
    users, _ = labels(dataset)
    (first,) = servable.recommend([servable.resolve(user_id=users[0], k=3)])
    banned = first[0]["item_id"]
    (second,) = servable.recommend(
        [servable.resolve(user_id=users[0], k=3, exclude=[banned])]
    )
    assert banned not in [entry["item_id"] for entry in second]


@pytest.mark.parametrize("k", [0, 101])
def test_k_outside_the_allowed_range_is_refused(
    tmp_path: Path, dataset: Dataset, k: int
):
    servable = served(tmp_path, "BPR", dataset)
    users, _ = labels(dataset)
    with pytest.raises(ServingError) as error:
        servable.resolve(user_id=users[0], k=k)
    assert error.value.status == 422


def test_an_unknown_user_is_a_404_by_default(tmp_path: Path, dataset: Dataset):
    with pytest.raises(ServingError) as error:
        served(tmp_path, "BPR", dataset).resolve(user_id="nobody")
    assert error.value.status == 404


def test_an_unknown_user_can_fall_back_to_popularity(tmp_path: Path, dataset: Dataset):
    servable = served(tmp_path, "BPR", dataset, unknown_user="popular")
    query = servable.resolve(user_id="nobody", k=3)
    assert query.fallback
    (answer,) = servable.recommend([query])
    popularity = torch.tensor((dataset.train_set.get_sparse() != 0).sum(axis=0)).ravel()
    _, items = labels(dataset)
    assert [e["item_id"] for e in answer] == [
        items[i] for i in torch.topk(popularity.float(), 3).indices.tolist()
    ]


def test_a_sequential_model_reads_the_stored_history(tmp_path: Path, dataset: Dataset):
    servable = served(tmp_path, "SASRec", dataset, mask_seen=False)
    users, items = labels(dataset)
    by_user = servable.resolve(user_id=users[2], k=5)
    history = [items[i] for i in by_user.history]
    by_history = servable.resolve(history=history, k=5)
    assert servable.recommend([by_user]) == servable.recommend([by_history])


def test_a_history_is_refused_by_a_general_model(tmp_path: Path, dataset: Dataset):
    _, items = labels(dataset)
    with pytest.raises(ServingError, match="not sequential"):
        served(tmp_path, "BPR", dataset).resolve(history=[items[0]])


def test_an_anonymous_history_is_refused_when_the_model_needs_the_user(
    tmp_path: Path, dataset: Dataset
):
    _, items = labels(dataset)
    servable = served(tmp_path, "Caser", dataset)
    with pytest.raises(ServingError, match="known user_id"):
        servable.resolve(history=[items[0], items[1]])


@pytest.mark.parametrize("history", [[], ["not-an-item"]])
def test_a_bad_history_is_refused(tmp_path: Path, dataset: Dataset, history):
    with pytest.raises(ServingError) as error:
        served(tmp_path, "SASRec", dataset).resolve(history=history)
    assert error.value.status == 422


def test_a_mixed_batch_answers_each_query_as_if_alone(tmp_path: Path, dataset: Dataset):
    servable = served(tmp_path, "SASRec", dataset, unknown_user="popular")
    users, items = labels(dataset)
    queries = [
        servable.resolve(user_id=users[0], k=3),
        servable.resolve(user_id="nobody", k=4),
        servable.resolve(history=[items[1], items[2], items[3]], k=5),
        servable.resolve(user_id=users[7], k=2, exclude=[items[0]]),
    ]
    together = servable.recommend(queries)
    alone = [servable.recommend([query])[0] for query in queries]
    # A batched matrix product may round differently from a single row, so the
    # ranking must match exactly and the scores only to float precision.
    assert ids(together) == ids(alone)
    assert scores(together) == pytest.approx(scores(alone), rel=1e-5)
    assert [len(answer) for answer in together] == [3, 4, 5, 2]


def test_an_old_checkpoint_serves_without_masking(tmp_path: Path, dataset: Dataset):
    state = make_model("BPR", dataset).get_state()  # no serving payload
    path = tmp_path / "old.pth"
    torch.save(state, path)
    servable = ServableModel.from_checkpoint(path)
    users, _ = labels(dataset)
    (answer,) = servable.recommend([servable.resolve(user_id=users[0], k=5)])
    assert [e["item_id"] for e in answer] == reference(
        make_model("BPR", dataset), dataset, 0, 5, mask=False
    )


def test_popular_fallback_needs_the_seen_matrix(tmp_path: Path, dataset: Dataset):
    path = tmp_path / "old.pth"
    torch.save(make_model("BPR", dataset).get_state(), path)
    with pytest.raises(ValueError, match="popular"):
        ServableModel.from_checkpoint(
            path, policy=ServingPolicy(unknown_user="popular")
        )


def test_context_aware_models_are_refused(tmp_path: Path, dataset: Dataset):
    path = tmp_path / "fm.pth"
    torch.save(make_model("FM", dataset).get_state(), path)
    with pytest.raises(ValueError, match="context-aware"):
        ServableModel.from_checkpoint(path)


def test_describe_reports_the_model(tmp_path: Path, dataset: Dataset):
    description = served(tmp_path, "SASRec", dataset).describe()
    assert description["model"] == "SASRec"
    assert description["kind"] == "sequential"
    assert description["n_items"] == dataset.info()["n_items"]
    assert description["needs_user"] is False
    assert isinstance(description["params"], dict)


def catalogue(tmp_path: Path, dataset: Dataset) -> dict:
    _, items = labels(dataset)
    path = tmp_path / "items.dat"
    path.write_text(
        "".join(f"{label}::Item {label}\n" for label in items.values()),
        encoding="utf-8",
    )
    return read_item_names(ItemMetadata(path=str(path), sep="::", header=False))


def test_names_are_read_by_position_without_a_header(tmp_path: Path, dataset: Dataset):
    names = catalogue(tmp_path, dataset)
    _, items = labels(dataset)
    assert names[str(items[0])] == f"Item {items[0]}"


def test_names_are_read_by_column_name_with_a_header(tmp_path: Path):
    path = tmp_path / "items.csv"
    path.write_text("title,movie_id\nHeat,7\n", encoding="utf-8")
    metadata = ItemMetadata(path=str(path), id_column="movie_id", name_column="title")
    assert read_item_names(metadata) == {"7": "Heat"}


def test_answers_carry_names_and_accept_them(tmp_path: Path, dataset: Dataset):
    path = save_servable(tmp_path / "m.pth", make_model("SASRec", dataset), dataset)
    servable = ServableModel.from_checkpoint(
        path, item_names=catalogue(tmp_path, dataset)
    )
    _, items = labels(dataset)
    by_id = servable.recommend([servable.resolve(history=[items[1], items[2]], k=3)])
    by_name = servable.recommend(
        [servable.resolve(history=[f"Item {items[1]}", f"Item {items[2]}"], k=3)]
    )
    assert by_id == by_name
    assert all(entry["name"] == f"Item {entry['item_id']}" for entry in by_id[0])


def test_scores_come_back_in_request_order_unmasked(tmp_path: Path, dataset: Dataset):
    servable = served(tmp_path, "BPR", dataset)
    model = make_model("BPR", dataset)
    model.eval()
    users, items = labels(dataset)
    wanted = [items[5], items[0], items[3]]
    scored = servable.score(items=wanted, user_id=users[1])
    with torch.inference_mode():
        expected = model.predict(user_indices=torch.tensor([1]))[0]
    assert [entry["item_id"] for entry in scored] == wanted
    assert [entry["score"] for entry in scored] == pytest.approx(
        [expected[i].item() for i in (5, 0, 3)]
    )


def test_scoring_an_unknown_item_is_refused(tmp_path: Path, dataset: Dataset):
    users, _ = labels(dataset)
    with pytest.raises(ServingError) as error:
        served(tmp_path, "BPR", dataset).score(items=["nope"], user_id=users[0])
    assert error.value.status == 422


def test_scoring_needs_at_least_one_item(tmp_path: Path, dataset: Dataset):
    users, _ = labels(dataset)
    with pytest.raises(ServingError, match="items"):
        served(tmp_path, "BPR", dataset).score(items=[], user_id=users[0])
