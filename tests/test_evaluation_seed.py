"""The configured 'evaluation.seed' decides the sampled candidates and the tie break.

The evaluator breaks ties from a generator seeded with its seed, and the
sampled loaders draw their negatives from it, but both default to 42. Built
without the configured value, every run used 42 whatever 'evaluation.seed'
said. The train pipeline also precomputes the evaluation loaders once before
its trials start, which only saves work if the trials ask for that same
sample again.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pandas as pd
import pytest
import torch
import yaml

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.common import dataset_preparation
from warprec.data.dataset import Dataset
from warprec.evaluation import build_evaluator
from warprec.evaluation.evaluator import Evaluator
from warprec.pipelines import design, estimate, eval as eval_module
from warprec.pipelines.remotes import ml
from warprec.recommenders.trainer import objectives
from warprec.utils import helpers
from warprec.utils.config.evaluation_configuration import EvaluationConfig
from warprec.utils.helpers import (
    build_evaluation_dataloader_kwargs,
    retrieve_evaluation_dataloader,
)

from conftest import CONTEXT_LABELS, N_ITEMS, make_model
from test_trial_optimization import RecordingTrainer, _bundle
from test_training_seed import _write_config

SEED = 11
SAMPLED: Dict[str, Any] = {
    "top_k": [5],
    "metrics": ["nDCG"],
    "strategy": "sampled",
    "num_negatives": 5,
    "seed": SEED,
}


@pytest.fixture
def seeds_seen(monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[Any]]:
    """Record the seed every evaluator and every evaluation loader is built with.

    Args:
        monkeypatch (pytest.MonkeyPatch): Wraps the evaluator and the loader
            helper in every module that builds them.

    Returns:
        Dict[str, List[Any]]: The evaluators' seeds and the loaders' seeds,
            None for a loader built without one.
    """
    seen: Dict[str, List[Any]] = {"evaluator": [], "loader": []}

    original_init = Evaluator.__init__

    def recording_init(self: Evaluator, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        seen["evaluator"].append(self.seed)

    def recording_loader(*args: Any, **kwargs: Any) -> Any:
        seen["loader"].append(kwargs.get("seed"))
        return retrieve_evaluation_dataloader(*args, **kwargs)

    monkeypatch.setattr(Evaluator, "__init__", recording_init)
    for module in (design, estimate, eval_module, ml, objectives):
        monkeypatch.setattr(module, "retrieve_evaluation_dataloader", recording_loader)
    return seen


def test_the_evaluator_is_built_with_the_configured_seed(dataset: Dataset):
    """build_evaluator hands the configured seed to the evaluator."""
    evaluator = build_evaluator(EvaluationConfig(**SAMPLED), dataset)

    assert evaluator.seed == SEED


def _all_tied(user_indices: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Score every item of every user alike.

    Args:
        user_indices (torch.Tensor): The batch of users.
        *args (Any): Ignored.
        **kwargs (Any): Ignored.

    Returns:
        torch.Tensor: Zeros, one row per user and one column per item.
    """
    return torch.zeros(len(user_indices), N_ITEMS)


def test_the_seed_decides_the_tie_break(dataset: Dataset):
    """A model that ties every item is ranked by the evaluation seed."""
    model = make_model("BPR", dataset)
    model.predict = _all_tied  # type: ignore[method-assign]

    def score(seed: int) -> float:
        """The nDCG@5 an all-tied model scores under one evaluation seed.

        Args:
            seed (int): The evaluation seed.

        Returns:
            float: The metric.
        """
        config = EvaluationConfig(top_k=[5], metrics=["nDCG"], seed=seed)
        evaluator = build_evaluator(config, dataset)
        evaluator.evaluate(
            model=model,
            dataloader=retrieve_evaluation_dataloader(
                dataset=dataset, model=model, strategy="full"
            ),
            strategy="full",
            dataset=dataset,
            device="cpu",
        )
        return float(torch.as_tensor(evaluator.compute_results()[5]["nDCG"]).mean())

    assert score(1) == score(1)
    assert score(1) != score(2)


def test_the_design_pipeline_evaluates_with_the_configured_seed(
    tmp_path: Path,
    interactions_frame: pd.DataFrame,
    seeds_seen: Dict[str, List[Any]],
):
    """The design pipeline's evaluator and loader both use evaluation.seed."""
    path = _write_config(tmp_path, interactions_frame)
    config = yaml.safe_load(Path(path).read_text())
    config["evaluation"] = SAMPLED
    Path(path).write_text(yaml.safe_dump(config))

    design.design_pipeline(path)

    assert seeds_seen["evaluator"] == [SEED]
    assert seeds_seen["loader"] == [SEED]


def test_the_eval_pipeline_evaluates_with_the_configured_seed(
    tmp_path: Path,
    interactions_frame: pd.DataFrame,
    seeds_seen: Dict[str, List[Any]],
):
    """The eval pipeline's evaluator and loader both use evaluation.seed."""
    path = _write_config(tmp_path, interactions_frame)
    config = yaml.safe_load(Path(path).read_text())
    config["evaluation"] = SAMPLED
    config["models"] = {"EASE": {"l2": 10.0, "optimization": {"num_workers": 0}}}
    config["writer"] = {
        "dataset_name": "seed",
        "writing_method": "local",
        "local_experiment_path": str(tmp_path),
    }
    Path(path).write_text(yaml.safe_dump(config))

    eval_module.eval_pipeline(path)

    assert seeds_seen["evaluator"] == [SEED]
    assert seeds_seen["loader"] == [SEED]


def test_the_estimate_pipeline_evaluates_with_the_configured_seed(
    tmp_path: Path,
    interactions_frame: pd.DataFrame,
    seeds_seen: Dict[str, List[Any]],
):
    """The estimate pipeline times the evaluation evaluation.seed asks for."""
    path = _write_config(
        tmp_path,
        interactions_frame,
        estimate={"warmup_batches": 0, "train_batches": 1, "eval_batches": 1},
        writer={
            "dataset_name": "seed",
            "writing_method": "local",
            "local_experiment_path": str(tmp_path),
        },
    )
    config = yaml.safe_load(Path(path).read_text())
    config["evaluation"] = SAMPLED
    Path(path).write_text(yaml.safe_dump(config))

    estimate.estimate_pipeline(path)

    assert seeds_seen["evaluator"] == [SEED]
    assert seeds_seen["loader"] == [SEED]


def test_a_trial_evaluates_with_the_configured_seed(
    monkeypatch: pytest.MonkeyPatch,
    dataset: Dataset,
    tmp_path: Path,
    seeds_seen: Dict[str, List[Any]],
):
    """A train-pipeline trial's evaluator and loader use evaluation.seed."""
    config = _bundle(monkeypatch, dataset, tmp_path)
    config["evaluation"] = EvaluationConfig(**SAMPLED, validation_metric="nDCG@5")
    config["strategy"] = "sampled"

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
    seeds_seen["evaluator"].clear()

    objectives.objective_function(config)

    assert seeds_seen["evaluator"] == [SEED]
    assert seeds_seen["loader"] == [SEED]


def test_the_final_evaluation_uses_the_configured_seed(
    dataset: Dataset, seeds_seen: Dict[str, List[Any]]
):
    """The train pipeline's final evaluation uses evaluation.seed."""
    ml.remote_evaluation_and_timing._function(  # type: ignore[attr-defined]
        model=make_model("EASE", dataset),
        main_dataset=dataset,
        evaluation=EvaluationConfig(**SAMPLED),
        num_workers=0,
        device="cpu",
        requires_timing=False,
        custom_modules=[],
    )

    assert seeds_seen["evaluator"] == [SEED]
    assert seeds_seen["loader"] == [SEED]


@pytest.mark.parametrize("model_name", ["BPR", "FM"])
def test_the_trials_reuse_the_precomputed_sample(
    interactions_frame: pd.DataFrame, model_name: str
):
    """The loader a trial asks for carries the sample prepared before the trials."""
    train = interactions_frame.groupby("user_id", group_keys=False).apply(
        lambda g: g.iloc[:-1]
    )
    evaluation = interactions_frame.groupby("user_id", group_keys=False).apply(
        lambda g: g.iloc[-1:]
    )
    fresh = Dataset(
        train_data=train,
        eval_data=evaluation,
        rating_type="implicit",
        timestamp_label="timestamp",
        context_labels=CONTEXT_LABELS,
        batch_size=64,
    )
    config = EvaluationConfig(
        **{**SAMPLED, "negative_sampling": "popularity", "neg_alpha": 0.5}
    )
    dataset_preparation(
        fresh, None, SimpleNamespace(evaluation=config, models={model_name: {}})
    )
    prepared = [loader.dataset for loader in fresh._precomputed_dataloader.values()]

    # Asked for the way a trial asks: its own worker settings, the configured
    # sampling, and the configured seed
    loader = helpers.retrieve_evaluation_dataloader(
        dataset=fresh,
        model=make_model(model_name, fresh),
        strategy="sampled",
        num_negatives=config.num_negatives,
        seed=config.seed,
        negative_sampling=config.negative_sampling,
        neg_alpha=config.neg_alpha,
        **build_evaluation_dataloader_kwargs(num_workers=0, device="cpu"),
    )

    assert len(prepared) == 1
    assert loader.dataset is prepared[0]
