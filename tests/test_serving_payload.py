"""A saved model carries what serving needs from the data it was trained on.

Serving loads a checkpoint and nothing else, so the items each user has seen -
to keep them out of recommendations - and the recent history of each user - to
feed a sequential model - have to travel inside it.
"""

from pathlib import Path

import numpy as np
import torch

from warprec.data.dataset import Dataset
from warprec.data.writer import LocalWriter
from warprec.serving.payload import build_serving_payload

from conftest import make_model


def test_the_seen_matrix_is_the_binary_training_matrix(dataset: Dataset):
    payload = build_serving_payload(make_model("BPR", dataset), dataset)
    train = dataset.train_set.get_sparse()
    assert payload["seen"].dtype == np.int8
    assert (payload["seen"] != (train != 0)).nnz == 0


def test_a_general_model_stores_no_histories(dataset: Dataset):
    assert (
        build_serving_payload(make_model("BPR", dataset), dataset)["histories"] is None
    )


def test_sequential_histories_are_the_most_recent_items_per_user(dataset: Dataset):
    model = make_model("SASRec", dataset)
    histories = build_serving_payload(model, dataset)["histories"]
    n_users = dataset.info()["n_users"]
    assert histories["offsets"].shape == (n_users + 1,)

    sequences, lengths = dataset.train_session.get_user_history_sequences(
        list(range(n_users)), model.max_seq_len
    )
    for user in range(n_users):
        start, end = histories["offsets"][user], histories["offsets"][user + 1]
        expected = sequences[user, : lengths[user]].tolist()
        assert histories["items"][start:end].tolist() == expected


def test_the_writer_embeds_the_payload_when_given_the_dataset(
    tmp_path: Path, dataset: Dataset
):
    model = make_model("LightGCN", dataset)
    writer = LocalWriter(dataset_name="payload", local_path=str(tmp_path))
    writer.write_model(model, dataset=dataset)

    (path,) = Path(writer.experiment_serialized_models_path).glob("*.pth")
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["serving"]["seen"].shape == (
        dataset.info()["n_users"],
        dataset.info()["n_items"],
    )
    assert "module" in state


def test_the_writer_still_works_without_the_dataset(tmp_path: Path, dataset: Dataset):
    writer = LocalWriter(dataset_name="nopayload", local_path=str(tmp_path))
    writer.write_model(make_model("BPR", dataset))

    (path,) = Path(writer.experiment_serialized_models_path).glob("*.pth")
    assert "serving" not in torch.load(path, map_location="cpu", weights_only=False)
