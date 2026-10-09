"""The configured seed reaches the loader every iterative model trains on.

The loaders take a seed for their shuffle order and their negative samples,
but it defaults to 42. A caller that builds the training loader without
passing the model's seed therefore trains every run on the same batches,
whatever 'optimization.properties.seed' says, and a seed sweep only varies
the weight initialisation.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import yaml

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.pipelines.design import design_pipeline
from warprec.pipelines.estimate import estimate_pipeline
from warprec.pipelines.eval import eval_pipeline
from warprec.pipelines.remotes.ml import remote_model_retraining
from warprec.recommenders.collaborative_filtering_recommender.latent_factor.bpr import (
    BPR,
)
from warprec.recommenders.trainer import objectives
from warprec.utils.registry import model_registry, params_registry

from conftest import make_model
from test_trial_optimization import RecordingTrainer, _bundle

SEED = 7
HYPERPARAMETERS = {
    "embedding_size": 8,
    "reg_weight": 1e-5,
    "batch_size": 64,
    "epochs": 1,
    "learning_rate": 0.001,
}


@pytest.fixture
def seeds_seen(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Record the seed every BPR training loader is built with.

    Args:
        monkeypatch (pytest.MonkeyPatch): Wraps BPR.get_dataloader.

    Returns:
        List[Any]: The seed of each loader built, None when none was passed.
    """
    seen: List[Any] = []
    original = BPR.get_dataloader

    def recording(self: BPR, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("seed"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BPR, "get_dataloader", recording)
    return seen


def _first_batch(model: Any, dataset: Dataset) -> List[np.ndarray]:
    """The first training batch of a model, built with the model's own seed.

    Args:
        model (Any): The iterative model.
        dataset (Dataset): The dataset it trains on.

    Returns:
        List[np.ndarray]: One array per tensor of the batch.
    """
    loader = model.get_dataloader(
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        seed=model.seed,
    )
    batch = next(iter(loader))
    parts = batch if isinstance(batch, (list, tuple)) else [batch]
    return [np.asarray(t) for t in parts]


@pytest.mark.parametrize("model_name", ["BPR", "NeuMF", "MultiVAE", "SASRec"])
def test_the_model_seed_decides_the_first_batch(dataset: Dataset, model_name: str):
    """Two seeds give different first batches; one seed gives the same one."""
    first = make_model(model_name, dataset)
    first.seed = 1
    again = make_model(model_name, dataset)
    again.seed = 1
    other = make_model(model_name, dataset)
    other.seed = 2

    a, b, c = (_first_batch(m, dataset) for m in (first, again, other))

    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    assert any(not np.array_equal(left, right) for left, right in zip(a, c))


def test_a_trial_trains_on_the_configured_seed(
    monkeypatch: pytest.MonkeyPatch,
    dataset: Dataset,
    tmp_path: Path,
    seeds_seen: List[Any],
):
    """A train-pipeline trial builds its loader with the configured seed."""
    config = _bundle(monkeypatch, dataset, tmp_path)
    config["seed"] = SEED

    monkeypatch.setattr(objectives.ray, "get", lambda value: value)
    monkeypatch.setattr(
        objectives.train,
        "get_context",
        lambda: SimpleNamespace(get_world_size=lambda: 1),
    )
    monkeypatch.setattr(objectives.train, "get_checkpoint", lambda: None)
    monkeypatch.setattr(objectives.train, "report", lambda *a, **k: None)
    monkeypatch.setattr(objectives.L, "Trainer", RecordingTrainer)
    monkeypatch.setattr(objectives, "RayTrainReportCallback", lambda: None)

    objectives.objective_function(config)

    assert seeds_seen == [SEED]


def test_the_retraining_trains_on_the_configured_seed(
    dataset: Dataset, seeds_seen: List[Any]
):
    """The model retrained after cross-validation uses the configured seed."""
    params = params_registry.get(
        "BPR", **HYPERPARAMETERS, optimization={"num_workers": 0}
    )
    remote_model_retraining._function(  # type: ignore[attr-defined]
        model_name="BPR",
        best_params={**HYPERPARAMETERS, "iterations": 1},
        main_dataset=dataset,
        params=params,
        custom_modules=[],
        device="cpu",
        seed=SEED,
    )

    assert seeds_seen == [SEED]


def _write_config(
    tmp_path: Path, interactions_frame: pd.DataFrame, **extra: Any
) -> str:
    """A pipeline configuration over the shared interactions, written to disk.

    Args:
        tmp_path (Path): Where to write the data and the configuration.
        interactions_frame (pd.DataFrame): The interactions to read.
        **extra (Any): Further top-level configuration blocks.

    Returns:
        str: The configuration's path.
    """
    data = tmp_path / "data.tsv"
    interactions_frame[["user_id", "item_id", "rating", "timestamp"]].to_csv(
        data, sep="\t", index=False
    )
    config: Dict[str, Any] = {
        "reader": {
            "loading_strategy": "dataset",
            "data_type": "transaction",
            "reading_method": "local",
            "local_path": str(data),
            "rating_type": "implicit",
            "sep": "\t",
            "labels": {
                "user_id_label": "user_id",
                "item_id_label": "item_id",
                "rating_label": "rating",
                "timestamp_label": "timestamp",
            },
        },
        "splitter": {"test_splitting": {"strategy": "temporal_holdout", "ratio": 0.2}},
        "models": {
            "BPR": {
                **HYPERPARAMETERS,
                "optimization": {"properties": {"seed": SEED}, "num_workers": 0},
            }
        },
        "evaluation": {"top_k": [5], "metrics": ["nDCG"], "strategy": "full"},
        "general": {"device": "cpu"},
        **extra,
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_the_design_pipeline_trains_on_the_configured_seed(
    tmp_path: Path, interactions_frame: pd.DataFrame, seeds_seen: List[Any]
):
    """The design pipeline builds its loader with the configured seed."""
    design_pipeline(_write_config(tmp_path, interactions_frame))

    assert seeds_seen == [SEED]


def test_the_estimate_pipeline_trains_on_the_configured_seed(
    tmp_path: Path, interactions_frame: pd.DataFrame, seeds_seen: List[Any]
):
    """The estimate pipeline times the loader the configured seed builds."""
    estimate_pipeline(
        _write_config(
            tmp_path,
            interactions_frame,
            estimate={"warmup_batches": 0, "train_batches": 1, "eval_batches": 1},
            writer={
                "dataset_name": "seed",
                "writing_method": "local",
                "local_experiment_path": str(tmp_path),
            },
        )
    )

    assert seeds_seen == [SEED]


def test_the_eval_pipeline_builds_models_with_the_configured_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, interactions_frame: pd.DataFrame
):
    """A model the eval pipeline builds carries its configured seed, not 42.

    It matters for whatever the model draws when it is built or asked to
    score, such as the Random baseline's ranking.
    """
    built: List[Any] = []
    original = model_registry.get

    def recording(*args: Any, **kwargs: Any) -> Any:
        model = original(*args, **kwargs)
        built.append(model.seed)
        return model

    monkeypatch.setattr(model_registry, "get", recording)

    path = _write_config(
        tmp_path,
        interactions_frame,
        writer={
            "dataset_name": "seed",
            "writing_method": "local",
            "local_experiment_path": str(tmp_path),
        },
    )
    config = yaml.safe_load(Path(path).read_text())
    optimization = {"properties": {"seed": SEED}, "num_workers": 0}
    config["models"] = {
        "Random": {"optimization": optimization},
        "EASE": {"l2": 10.0, "optimization": optimization},
    }
    Path(path).write_text(yaml.safe_dump(config))

    eval_pipeline(path)

    assert built == [SEED, SEED]
