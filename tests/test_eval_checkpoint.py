"""How the evaluation pipeline brings back a model saved by a training run.

A closed-form model keeps what it learned in plain attributes, so it has to be
restored from the checkpoint rather than refitted: a refit would evaluate a
different model, one fitted with whatever hyperparameters the evaluation
configuration happens to list. And a checkpoint only scores the right items on a
dataset whose user and item ids map to the same indices as the one it was
trained on; the same dimensions are not enough.
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
import torch

from warprec.data.dataset import Dataset
from warprec.pipelines.eval import _check_checkpoint_paths, _load_or_build_model

from conftest import build_params, make_model


def _save(model: Any, path: Path) -> str:
    """Write a checkpoint the way the train pipeline's writer does.

    Args:
        model (Any): The model to save.
        path (Path): Where to write it.

    Returns:
        str: The path written.
    """
    torch.save(model.get_state(), path)
    return str(path)


def _scores(model: Any, dataset: Dataset) -> torch.Tensor:
    """Every user's score for every item.

    Args:
        model (Any): The model to score with.
        dataset (Dataset): The dataset it is evaluated on.

    Returns:
        torch.Tensor: The scores.
    """
    model.eval()
    users = torch.arange(dataset.info()["n_users"])
    with torch.inference_mode():
        return model.predict(user_indices=users).float()


def _restore(name: str, params: Dict[str, Any], path: str, dataset: Dataset) -> Any:
    """Bring a model back the way the evaluation pipeline does.

    Args:
        name (str): The model name.
        params (Dict[str, Any]): Its configured hyperparameters.
        path (str): The checkpoint to load.
        dataset (Dataset): The dataset to evaluate on.

    Returns:
        Any: The model to evaluate.
    """
    return _load_or_build_model(
        model_name=name,
        model_params={**params, "meta": {"load_from": path}},
        dataset=dataset,
        block_size=50,
        chunk_size=4096,
    )


@pytest.fixture(name="relabelled")
def relabelled_fixture(interactions_frame: pd.DataFrame) -> Dataset:
    """The same interactions, but every user carries another id.

    The dimensions match the shared dataset exactly, so only the ids tell the
    two apart.

    Args:
        interactions_frame (pd.DataFrame): The generated interactions.

    Returns:
        Dataset: A dataset with the same shape and different user ids.
    """
    frame = interactions_frame.drop(columns=["daytime", "weather"]).copy()
    frame["user_id"] = frame["user_id"] + 1000
    train = frame.groupby("user_id", group_keys=False).apply(lambda g: g.iloc[:-1])
    evaluation = frame.groupby("user_id", group_keys=False).apply(lambda g: g.iloc[-1:])
    return Dataset(
        train_data=train,
        eval_data=evaluation,
        rating_type="explicit",
        rating_label="rating",
        timestamp_label="timestamp",
        batch_size=64,
    )


def test_a_closed_form_model_is_restored_not_refitted(dataset: Dataset, tmp_path):
    fitted = make_model("EASE", dataset)
    path = _save(fitted, tmp_path / "EASE.pth")

    # Configured differently from the checkpoint: a refit would follow this.
    restored = _restore("EASE", {**build_params("EASE"), "l2": 500.0}, path, dataset)

    assert torch.allclose(_scores(restored, dataset), _scores(fitted, dataset))
    assert restored.l2 == fitted.l2 != 500.0


def test_an_iterative_model_still_gets_its_weights(dataset: Dataset, tmp_path):
    trained = make_model("BPR", dataset, seed=1)
    path = _save(trained, tmp_path / "BPR.pth")

    restored = _restore("BPR", build_params("BPR"), path, dataset)

    assert torch.equal(_scores(restored, dataset), _scores(trained, dataset))


@pytest.mark.parametrize("model_name", ["EASE", "BPR"])
def test_a_missing_checkpoint_is_refused_for_any_model(model_name: str, tmp_path):
    missing = str(tmp_path / "nowhere.pth")
    with pytest.raises(FileNotFoundError, match="nowhere.pth"):
        _check_checkpoint_paths(
            {model_name: {**build_params(model_name), "meta": {"load_from": missing}}}
        )


def test_models_without_a_checkpoint_pass_the_check():
    _check_checkpoint_paths(
        {"EASE": build_params("EASE"), "BPR": {**build_params("BPR"), "meta": {}}}
    )


@pytest.mark.parametrize("model_name", ["EASE", "BPR"])
def test_a_checkpoint_from_other_ids_is_refused(
    model_name: str, dataset: Dataset, relabelled: Dataset, tmp_path
):
    assert relabelled.info()["n_users"] == dataset.info()["n_users"]
    assert relabelled.info()["n_items"] == dataset.info()["n_items"]
    path = _save(make_model(model_name, dataset), tmp_path / "model.pth")

    with pytest.raises(ValueError, match="user"):
        _restore(model_name, build_params(model_name), path, relabelled)


def test_a_checkpoint_of_another_model_is_refused(dataset: Dataset, tmp_path):
    path = _save(make_model("BPR", dataset), tmp_path / "BPR.pth")

    with pytest.raises(ValueError, match="BPR"):
        _restore("EASE", build_params("EASE"), path, dataset)
