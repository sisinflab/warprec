"""A served model answers exactly what the same model would compute offline.

Every test here compares the serving path against a direct call to the model,
so that ID mapping, masking, padding and batching can only pass by agreeing
with the model itself.
"""

import math

import numpy as np
from pathlib import Path
from typing import List

import pytest
import torch

from warprec.data.dataset import Dataset
from warprec.recommenders.base_recommender import Recommender
from warprec.serving.servable import (
    Presentation,
    ServableModel,
    ServingError,
    ServingPolicy,
)

from warprec.serving.catalogue import Catalogue, read_catalogue
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


def test_describe_reports_the_model(tmp_path: Path, dataset: Dataset):
    description = served(tmp_path, "SASRec", dataset).describe()
    assert description["model"] == "SASRec"
    assert description["kind"] == "sequential"
    assert description["n_items"] == dataset.info()["n_items"]
    assert description["needs_user"] is False
    assert isinstance(description["params"], dict)


TITLES = {0: "Toy Story (1995)", 1: "Toy Story 2 (1999)", 2: "Heat (1995)"}
GENRES = ["Comedy", "Drama|Comedy", "Action|Crime"]


def catalogue(tmp_path: Path, dataset: Dataset) -> Catalogue:
    """A catalogue naming every item, a few like real films, with their genres."""
    _, items = labels(dataset)
    path = tmp_path / "items.dat"
    path.write_text(
        "".join(
            f"{label}::{TITLES.get(index, f'Item {label}')}::{GENRES[index % 3]}\n"
            for index, label in items.items()
        ),
        encoding="utf-8",
    )
    metadata = ItemMetadata(
        path=str(path),
        sep="::",
        header=False,
        columns={"genres": {"column": 2, "separator": "|"}},
    )
    return read_catalogue(metadata)


def with_catalogue(
    tmp_path: Path, dataset: Dataset, name: str = "SASRec", **policy
) -> ServableModel:
    path = save_servable(tmp_path / f"{name}.pth", make_model(name, dataset), dataset)
    return ServableModel.from_checkpoint(
        path, policy=ServingPolicy(**policy), catalogue=catalogue(tmp_path, dataset)
    )


def test_names_and_attributes_are_read_by_position(tmp_path: Path, dataset: Dataset):
    read = catalogue(tmp_path, dataset)
    _, items = labels(dataset)
    assert read.names[str(items[3])] == f"Item {items[3]}"
    assert read.attributes[str(items[1])] == {"genres": ["Drama", "Comedy"]}


def test_names_are_read_by_column_name_with_a_header(tmp_path: Path):
    path = tmp_path / "items.csv"
    path.write_text("title,movie_id,year\nHeat,7,1995\n", encoding="utf-8")
    metadata = ItemMetadata(
        path=str(path),
        id_column="movie_id",
        name_column="title",
        columns={"year": "year"},
    )
    read = read_catalogue(metadata)
    assert read.names == {"7": "Heat"}
    assert read.attributes == {"7": {"year": "1995"}}


def test_answers_carry_names_and_attributes_and_accept_names(
    tmp_path: Path, dataset: Dataset
):
    servable = with_catalogue(tmp_path, dataset)
    _, items = labels(dataset)
    by_id = servable.recommend([servable.resolve(history=[items[1], items[2]], k=3)])
    by_name = servable.recommend(
        [servable.resolve(history=[TITLES[1], TITLES[2]], k=3)]
    )
    assert by_id == by_name
    read = catalogue(tmp_path, dataset)
    for entry in by_id[0]:
        assert entry["name"] == read.names[str(entry["item_id"])]
        assert entry["attributes"] == read.attributes[str(entry["item_id"])]


def test_names_are_matched_whatever_their_case(tmp_path: Path, dataset: Dataset):
    servable = with_catalogue(tmp_path, dataset)
    _, items = labels(dataset)
    exact = servable.resolve(history=[items[0]], k=3)
    folded = servable.resolve(history=["TOY STORY (1995)"], k=3)
    assert exact.history == folded.history


def test_an_unknown_name_suggests_the_closest(tmp_path: Path, dataset: Dataset):
    servable = with_catalogue(tmp_path, dataset)
    with pytest.raises(
        ServingError, match=r"Did you mean.*Toy Story \(1995\)"
    ) as error:
        servable.resolve(history=["Toy Stroy (1995)"])
    assert error.value.status == 422


def test_search_finds_items_by_part_of_their_name(tmp_path: Path, dataset: Dataset):
    servable = with_catalogue(tmp_path, dataset)
    found = servable.search_items("toy story")
    assert {entry["name"] for entry in found["matches"]} == {TITLES[0], TITLES[1]}
    seen = dataset.train_set.get_sparse()
    _, items = labels(dataset)
    index = {str(label): i for i, label in items.items()}
    for entry in found["matches"]:
        assert (
            entry["interactions"] == (seen[:, index[str(entry["item_id"])]] != 0).sum()
        )
        assert "genres" in entry["attributes"]


def test_search_ranks_an_exact_name_first(tmp_path: Path, dataset: Dataset):
    found = with_catalogue(tmp_path, dataset).search_items("toy story (1995)")
    assert found["matches"][0]["name"] == TITLES[0]


def test_search_respects_its_limit(tmp_path: Path, dataset: Dataset):
    assert (
        len(with_catalogue(tmp_path, dataset).search_items("item", limit=4)["matches"])
        == 4
    )


def test_search_says_when_an_item_is_unknown(tmp_path: Path, dataset: Dataset):
    """'Are you trained on Kung Fu Panda?' gets a no, not a guess."""
    servable = with_catalogue(tmp_path, dataset)
    assert servable.search_items("Kung Fu Panda")["matches"] == []
    near = servable.search_items("Toy Stroy (1995)")
    assert near["matches"] == []
    assert TITLES[0] in [entry["name"] for entry in near["suggestions"]]


def test_search_needs_a_catalogue(tmp_path: Path, dataset: Dataset):
    with pytest.raises(ServingError, match="catalogue") as error:
        served(tmp_path, "BPR", dataset).search_items("toy")
    assert error.value.status == 422


def test_items_are_looked_up_by_id_or_name(tmp_path: Path, dataset: Dataset):
    servable = with_catalogue(tmp_path, dataset)
    _, items = labels(dataset)
    found = servable.get_items([items[0], "heat (1995)", "Toy Stroy (1995)"])
    assert [entry["name"] for entry in found["items"]] == [TITLES[0], TITLES[2]]
    (unknown,) = found["unknown"]
    assert (
        unknown["query"] == "Toy Stroy (1995)" and TITLES[0] in unknown["suggestions"]
    )


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


@pytest.mark.parametrize(
    "unknown_user, expected", [("popular", "fallback"), ("error", 404)]
)
def test_a_user_with_no_training_history_is_treated_as_unknown(
    dataset: Dataset, unknown_user: str, expected
):
    """A cold-start protocol keeps held-out users in the mapping with no
    training rows; their embedding never trained, so it must not answer."""
    from scipy.sparse import csr_matrix

    from warprec.serving.payload import build_serving_payload

    model = make_model("BPR", dataset)
    seen = build_serving_payload(model, dataset)["seen"].tolil()
    seen.rows[0], seen.data[0] = [], []
    servable = ServableModel(
        model,
        payload={"seen": csr_matrix(seen)},
        policy=ServingPolicy(unknown_user=unknown_user),
    )
    users, _ = labels(dataset)
    if expected == "fallback":
        assert servable.resolve(user_id=users[0]).fallback
        assert not servable.resolve(user_id=users[1]).fallback
    else:
        with pytest.raises(ServingError) as error:
            servable.resolve(user_id=users[0])
        assert error.value.status == 404


@pytest.mark.parametrize(
    "name, expected",
    [("EASE", False), ("ItemKNN", False), ("BPR", True), ("LightGCN", True)],
)
def test_a_checkpoint_says_whether_its_model_can_use_a_gpu(
    tmp_path: Path, dataset: Dataset, name: str, expected: bool
):
    """Closed-form models keep what they learned in numpy arrays and score on
    the CPU wherever they are placed, so a GPU would sit idle."""
    from warprec.serving.servable import scores_on_device

    path = save_servable(tmp_path / f"{name}.pth", make_model(name, dataset), dataset)
    assert scores_on_device(path) is expected


def test_a_batch_of_score_requests_answers_each_as_if_alone(
    tmp_path: Path, dataset: Dataset
):
    """Scoring is batched like recommending: one forward pass for the batch,
    then every request reads its own candidates, in its own order."""
    servable = served(tmp_path, "SASRec", dataset, unknown_user="popular")
    users, items = labels(dataset)
    queries = [
        servable.resolve_scoring(items=[items[3], items[1]], user_id=users[0]),
        servable.resolve_scoring(items=[items[2]], user_id="nobody"),
        servable.resolve_scoring(
            items=[items[4], items[0], items[5]], history=[items[1], items[2]]
        ),
    ]
    together = servable.score_batch(queries)
    alone = [servable.score_batch([query])[0] for query in queries]
    assert (
        ids(together)
        == ids(alone)
        == [[items[3], items[1]], [items[2]], [items[4], items[0], items[5]]]
    )
    assert scores(together) == pytest.approx(scores(alone), rel=1e-5)


@pytest.mark.parametrize("name", ["BPR", "SASRec", "Caser", "EASE"])
def test_the_example_request_of_a_card_is_accepted(
    tmp_path: Path, dataset: Dataset, name: str
):
    """What the card tells a client to send must work when it is sent."""
    servable = served(tmp_path, name, dataset)
    card = servable.describe()
    assert any("user_id" in sentence for sentence in card["how_to_ask"])
    (answer,) = servable.recommend([servable.resolve(**card["example_request"])])
    assert answer


def test_a_card_speaks_of_the_items_as_configured(tmp_path: Path, dataset: Dataset):
    path = save_servable(tmp_path / "m.pth", make_model("SASRec", dataset), dataset)
    presentation = Presentation(
        description="Films from a toy dataset", item_noun="movie"
    )
    card = ServableModel.from_checkpoint(path, presentation=presentation).describe()
    assert card["description"] == "Films from a toy dataset"
    assert card["item_noun"] == "movie"
    assert any("movies" in sentence for sentence in card["how_to_ask"])


def test_a_card_describes_the_catalogue_and_the_training(
    tmp_path: Path, dataset: Dataset
):
    card = with_catalogue(tmp_path, dataset).describe()
    assert card["catalogue"]["names"] is True
    assert card["catalogue"]["n_items"] == dataset.info()["n_items"]
    genres = card["catalogue"]["attributes"]["genres"]
    assert set(genres["examples"]) <= {"Comedy", "Drama", "Action", "Crime"}
    assert (
        card["training"]["n_interactions"] == (dataset.train_set.get_sparse() != 0).nnz
    )


def test_a_card_without_catalogue_says_so(tmp_path: Path, dataset: Dataset):
    card = served(tmp_path, "BPR", dataset).describe()
    assert card["catalogue"] == {
        "names": False,
        "n_items": dataset.info()["n_items"],
        "attributes": {},
    }


def test_the_example_user_is_one_the_model_learned_from(dataset: Dataset):
    """A user held out of training would be refused, so the card never offers one."""
    from scipy.sparse import csr_matrix

    from warprec.serving.payload import build_serving_payload

    model = make_model("BPR", dataset)
    seen = build_serving_payload(model, dataset)["seen"].tolil()
    seen.rows[0], seen.data[0] = [], []
    servable = ServableModel(model, payload={"seen": csr_matrix(seen)})
    users, _ = labels(dataset)
    example = servable.describe()["example_request"]
    assert example["user_id"] != users[0]
    servable.recommend([servable.resolve(**example)])


def genre_index(dataset: Dataset):
    """Internal item index -> its genres, as the test catalogue assigns them."""
    _, items = labels(dataset)
    return {index: GENRES[index % 3].split("|") for index in items}


def test_a_filter_keeps_only_matching_items(tmp_path: Path, dataset: Dataset):
    """'Only comedies' ranks exactly as the model would among comedies alone."""
    servable = with_catalogue(tmp_path, dataset, "BPR")
    model = make_model("BPR", dataset)
    model.eval()
    users, items = labels(dataset)
    comedies = [i for i, genres in genre_index(dataset).items() if "Comedy" in genres]
    (answer,) = servable.recommend(
        [servable.resolve(user_id=users[0], k=5, filter={"genres": "comedy"})]
    )
    with torch.inference_mode():
        scores = model.predict(user_indices=torch.tensor([0]))[0]
    scores = scores.clone()
    scores[dataset.train_set.get_sparse()[0].indices] = -math.inf
    blocked = torch.ones_like(scores, dtype=torch.bool)
    blocked[comedies] = False
    scores[blocked] = -math.inf
    expected = [
        items[i]
        for i in torch.topk(scores, 5).indices.tolist()
        if math.isfinite(scores[i])
    ]
    assert [entry["item_id"] for entry in answer] == expected
    assert all("Comedy" in entry["attributes"]["genres"] for entry in answer)


def test_a_filter_value_list_matches_any_of_them(tmp_path: Path, dataset: Dataset):
    servable = with_catalogue(tmp_path, dataset, "BPR", max_k=1000)
    users, _ = labels(dataset)
    (answer,) = servable.recommend(
        [
            servable.resolve(
                user_id=users[0], k=1000, filter={"genres": ["Action", "Drama"]}
            )
        ]
    )
    assert answer and all(
        {"Action", "Drama"} & set(e["attributes"]["genres"]) for e in answer
    )


@pytest.mark.parametrize(
    "filter_, message",
    [({"mood": "happy"}, "genres"), ({"genres": "Comdy"}, "Did you mean.*Comedy")],
)
def test_a_bad_filter_says_what_it_accepts(
    tmp_path: Path, dataset: Dataset, filter_, message
):
    servable = with_catalogue(tmp_path, dataset, "BPR")
    users, _ = labels(dataset)
    with pytest.raises(ServingError, match=message) as error:
        servable.resolve(user_id=users[0], filter=filter_)
    assert error.value.status == 422


def test_a_filter_needs_a_catalogue(tmp_path: Path, dataset: Dataset):
    users, _ = labels(dataset)
    with pytest.raises(ServingError, match="catalogue"):
        served(tmp_path, "BPR", dataset).resolve(
            user_id=users[0], filter={"genres": "Comedy"}
        )


def test_popular_items_follow_the_training_interactions(
    tmp_path: Path, dataset: Dataset
):
    servable = with_catalogue(tmp_path, dataset, "BPR")
    _, items = labels(dataset)
    counts = torch.tensor(
        np.asarray((dataset.train_set.get_sparse() != 0).sum(axis=0)).ravel()
    ).float()
    expected = [items[i] for i in torch.topk(counts, 4).indices.tolist()]
    answer = servable.popular_items(k=4)
    assert [entry["item_id"] for entry in answer] == expected
    assert [entry["interactions"] for entry in answer] == sorted(
        (entry["interactions"] for entry in answer), reverse=True
    )
    comedies = servable.popular_items(
        k=4, filter={"genres": "Comedy"}, exclude=[expected[0]]
    )
    assert all("Comedy" in e["attributes"]["genres"] for e in comedies)
    assert expected[0] not in [e["item_id"] for e in comedies]


def test_popular_items_need_the_training_interactions(tmp_path: Path, dataset: Dataset):
    torch.save(make_model("BPR", dataset).get_state(), tmp_path / "old.pth")
    with pytest.raises(ServingError, match="interactions"):
        ServableModel.from_checkpoint(tmp_path / "old.pth").popular_items()


def test_explanations_are_training_co_occurrences(tmp_path: Path, dataset: Dataset):
    """'Because' names the user's own items most often consumed with each one."""
    servable = served(tmp_path, "BPR", dataset)
    users, items = labels(dataset)
    seen = (dataset.train_set.get_sparse() != 0).astype(np.int64).tocsc()
    history = set(dataset.train_set.get_sparse()[0].indices.tolist())
    index = {str(label): i for i, label in items.items()}
    (answer,) = servable.recommend(
        [servable.resolve(user_id=users[0], k=3, explain=True)]
    )
    for entry in answer:
        target = index[str(entry["item_id"])]
        counts = {h: int(seen[:, h].multiply(seen[:, target]).sum()) for h in history}
        best = sorted((c for c in counts.values() if c > 0), reverse=True)[:2]
        assert [e["co_occurrences"] for e in entry["because"]] == best
        assert all(index[str(e["item_id"])] in history for e in entry["because"])


@pytest.mark.parametrize("together", [300, 386])
def test_explanations_count_beyond_a_byte(dataset: Dataset, together: int):
    """The seen matrix is stored one byte per cell; co-occurrences above 127
    must still be counted, never wrapped (386 used to become -126 and vanish)."""
    from scipy.sparse import csr_matrix, vstack

    from warprec.serving.payload import build_serving_payload

    model = make_model("BPR", dataset)
    stored = build_serving_payload(model, dataset)["seen"]
    n_items = stored.shape[1]
    # User 0 saw item 0 only; every other row saw every item, so item 0 was
    # consumed with each candidate by exactly `together` users.
    first = np.zeros((1, n_items), dtype=np.int8)
    first[0, 0] = 1
    others = np.ones((together, n_items), dtype=np.int8)
    seen = vstack([csr_matrix(first), csr_matrix(others)]).tocsr().astype(stored.dtype)
    servable = ServableModel(model, payload={"seen": seen})
    users, items = labels(dataset)
    (answer,) = servable.recommend(
        [servable.resolve(user_id=users[0], k=3, explain=True)]
    )
    assert answer
    for entry in answer:
        assert [(str(e["item_id"]), e["co_occurrences"]) for e in entry["because"]] == [
            (str(items[0]), together)
        ]
    (most_popular,) = servable.popular_items(k=1)
    assert most_popular["interactions"] == together + 1


def test_a_session_is_explained_from_its_own_items(tmp_path: Path, dataset: Dataset):
    servable = served(tmp_path, "SASRec", dataset)
    _, items = labels(dataset)
    session = [items[1], items[2], items[3]]
    (answer,) = servable.recommend(
        [servable.resolve(history=session, k=3, explain=True)]
    )
    assert all(e["item_id"] in session for entry in answer for e in entry["because"])


def test_no_explanation_unless_asked(tmp_path: Path, dataset: Dataset):
    users, _ = labels(dataset)
    servable = served(tmp_path, "BPR", dataset)
    (answer,) = servable.recommend([servable.resolve(user_id=users[0], k=3)])
    assert all("because" not in entry for entry in answer)


def test_the_card_mentions_filters_and_explanations(tmp_path: Path, dataset: Dataset):
    sentences = " ".join(
        with_catalogue(tmp_path, dataset, "BPR").describe()["how_to_ask"]
    )
    assert "filter" in sentences and "genres" in sentences and "explain" in sentences
