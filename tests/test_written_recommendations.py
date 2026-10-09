"""The recommendations a run writes are the lists it evaluated.

A model that ties many items is ranked by the tie break. The evaluator breaks
ties per user from 'evaluation.seed'; a writer that took ``torch.topk`` instead
wrote the lowest item ids, a list the reported numbers never described. Both
now draw the same keys for a user and an item, whichever batch the user falls
in, so the two lists agree item for item.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest
import torch
import yaml

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.data.writer import LocalWriter
from warprec.evaluation import build_evaluator
from warprec.evaluation import evaluator as evaluator_module
from warprec.pipelines.eval import eval_pipeline
from warprec.pipelines.remotes.data import remote_generate_recs
from warprec.recommenders.reranking import build_reranker
from warprec.utils.config import RerankConfig
from warprec.utils.config.evaluation_configuration import EvaluationConfig
from warprec.utils.config.writer_configuration import RecommendationWriting
from warprec.utils.helpers import retrieve_evaluation_dataloader

from conftest import make_model
from test_training_seed import _write_config

K = 5
SEED = 3


def _tied_model(dataset: Dataset) -> Any:
    """A model that scores every item of every user alike.

    Args:
        dataset (Dataset): The dataset to build it on.

    Returns:
        Any: The model, its predict replaced by a constant.
    """
    model = make_model("BPR", dataset)
    n_items = dataset.info()["n_items"]

    def predict(user_indices: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Score every item zero.

        Args:
            user_indices (torch.Tensor): The batch of users.
            *args (Any): Ignored.
            **kwargs (Any): Ignored.

        Returns:
            torch.Tensor: Zeros, [batch, items].
        """
        return torch.zeros(len(user_indices), n_items)

    model.predict = predict  # type: ignore[method-assign]
    return model


def _evaluated_lists(
    monkeypatch: pytest.MonkeyPatch,
    model: Any,
    dataset: Dataset,
    reranker: Optional[Any],
) -> Dict[int, List[int]]:
    """The top-k list the evaluator ranked for each user it evaluated.

    Args:
        monkeypatch (pytest.MonkeyPatch): Records what the evaluator ranked.
        model (Any): The model to evaluate.
        dataset (Dataset): The dataset to evaluate on.
        reranker (Optional[Any]): The re-ranker, if any.

    Returns:
        Dict[int, List[int]]: Item indices, in order, per user index.
    """
    ranked: Dict[int, List[int]] = {}
    current: Dict[str, torch.Tensor] = {}
    step = evaluator_module.Evaluator._compute_metrics_step
    rank = evaluator_module.top_k_breaking_ties

    def keep(indices: torch.Tensor) -> None:
        """File the first K items of each row under the batch's users.

        Args:
            indices (torch.Tensor): The ranked item indices of the batch.
        """
        for user, row in zip(current["users"].tolist(), indices[:, :K].tolist()):
            ranked[user] = row

    def recording_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        current["users"] = (
            kwargs["user_indices"] if "user_indices" in kwargs else args[2]
        )
        return step(self, *args, **kwargs)

    def recording_rank(*args: Any, **kwargs: Any) -> Any:
        values, indices = rank(*args, **kwargs)
        keep(indices)
        return values, indices

    monkeypatch.setattr(
        evaluator_module.Evaluator, "_compute_metrics_step", recording_step
    )
    monkeypatch.setattr(evaluator_module, "top_k_breaking_ties", recording_rank)
    if reranker is not None:
        call = type(reranker).__call__

        def recording_rerank(self: Any, *args: Any, **kwargs: Any) -> Any:
            values, indices = call(self, *args, **kwargs)
            keep(indices)
            return values, indices

        monkeypatch.setattr(type(reranker), "__call__", recording_rerank)

    config = EvaluationConfig(top_k=[K], metrics=["nDCG"], mask_seen="pair", seed=SEED)
    evaluator = build_evaluator(config, dataset, reranker)
    evaluator.evaluate(
        model=model,
        dataloader=retrieve_evaluation_dataloader(
            dataset=dataset, model=model, strategy="full"
        ),
        strategy="full",
        dataset=dataset,
        device="cpu",
    )
    return ranked


def _written_lists(
    tmp_path: Path, model: Any, dataset: Dataset, reranker: Optional[Any]
) -> Dict[int, List[int]]:
    """The top-k list the writer wrote for each user.

    Args:
        tmp_path (Path): Where to write.
        model (Any): The model.
        dataset (Dataset): The dataset.
        reranker (Optional[Any]): The re-ranker, if any.

    Returns:
        Dict[int, List[int]]: Item indices, in order, per user index.
    """
    writer = LocalWriter(dataset_name="recs", local_path=str(tmp_path))
    writer.write_recs(model=model, dataset=dataset, k=K, reranker=reranker, seed=SEED)
    written = pd.read_csv(
        next(Path(writer.experiment_recommendation_path).glob("*.tsv")), sep="\t"
    )
    users = dataset.info()["user_mapping"]
    items = dataset.info()["item_mapping"]
    lists: Dict[int, List[int]] = {}
    for user, item in zip(written["user_id"], written["item_id"]):
        lists.setdefault(users[user], []).append(items[item])
    return lists


@pytest.mark.parametrize("rerank", [None, "MMR"])
def test_a_tied_model_writes_the_lists_it_was_evaluated_on(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset: Dataset,
    rerank: Optional[str],
):
    """Every evaluated user's written list is the evaluator's top-k, in order."""
    model = _tied_model(dataset)
    reranker = (
        build_reranker(RerankConfig(name=rerank, pool=10), dataset) if rerank else None
    )

    evaluated = _evaluated_lists(monkeypatch, model, dataset, reranker)
    written = _written_lists(tmp_path, model, dataset, reranker)

    assert evaluated, "the evaluator ranked nobody"
    for user, ranking in evaluated.items():
        assert written[user] == ranking, f"user {user} was written another list"

    # And the lists are drawn, not the lowest ids for everyone
    assert len({tuple(row) for row in written.values()}) > 1


@pytest.fixture
def seeds_written(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Record the seed every recommendation file is written with.

    Args:
        monkeypatch (pytest.MonkeyPatch): Wraps LocalWriter.write_recs.

    Returns:
        List[Any]: The seed of each call, None when none was passed.
    """
    seen: List[Any] = []
    original = LocalWriter.write_recs

    def recording(self: LocalWriter, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("seed"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LocalWriter, "write_recs", recording)
    return seen


def test_the_eval_pipeline_writes_with_the_evaluation_seed(
    tmp_path: Path, interactions_frame: pd.DataFrame, seeds_written: List[Any]
):
    """The eval pipeline's recommendation file is drawn from evaluation.seed."""
    path = _write_config(tmp_path, interactions_frame)
    config = yaml.safe_load(Path(path).read_text())
    config["evaluation"]["seed"] = SEED
    config["models"] = {
        "EASE": {
            "l2": 10.0,
            "meta": {"save_recs": True},
            "optimization": {"num_workers": 0},
        }
    }
    config["writer"] = {
        "dataset_name": "seed",
        "writing_method": "local",
        "local_experiment_path": str(tmp_path),
        "recommendation": {"k": K},
    }
    Path(path).write_text(yaml.safe_dump(config))

    eval_pipeline(path)

    assert seeds_written == [SEED]


def test_the_train_pipeline_writes_with_the_evaluation_seed(
    tmp_path: Path, dataset: Dataset, seeds_written: List[Any]
):
    """The train pipeline's recommendation task passes evaluation.seed on."""
    config = SimpleNamespace(
        general=SimpleNamespace(custom_modules=[]),
        rerank=RerankConfig(),
        evaluation=EvaluationConfig(top_k=[K], metrics=["nDCG"], seed=SEED),
        writer=SimpleNamespace(recommendation=RecommendationWriting(k=K)),
    )
    remote_generate_recs._function(  # type: ignore[attr-defined]
        writer=LocalWriter(dataset_name="recs", local_path=str(tmp_path)),
        model=make_model("EASE", dataset),
        dataset=dataset,
        config=config,
        device="cpu",
    )

    assert seeds_written == [SEED]
