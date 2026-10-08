"""The estimate pipeline times evaluation with the evaluator's own batches.

It runs the evaluation batches itself, to time each one, so it has to read
every batch layout the evaluation loaders produce: the full loader yields a
sparse ground truth, the sampled one positives and negatives, the contextual
ones a context per row. Whatever it measures, the metrics it computes along the
way have to be the ones the evaluator computes on the same batches.
"""

from typing import Any, Dict

import pytest
import torch

from warprec.data.dataset import Dataset
from warprec.evaluation import build_evaluator
from warprec.pipelines.estimate import EstimateStageTracker, _estimate_eval_loop
from warprec.utils.config.evaluation_configuration import EvaluationConfig
from warprec.utils.helpers import retrieve_evaluation_dataloader

from conftest import make_model


def _config() -> EvaluationConfig:
    """The evaluation the test measures.

    Returns:
        EvaluationConfig: Two metrics at two cutoffs.
    """
    return EvaluationConfig(
        top_k=[3, 5],
        metrics=["nDCG", "Recall"],
        validation_metric="nDCG@3",
        batch_size=16,
        num_negatives=5,
    )


def _as_floats(results: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
    """Flatten an evaluator's results to one number per metric and cutoff.

    Args:
        results (Dict[int, Dict[str, Any]]): The evaluator's results.

    Returns:
        Dict[str, float]: The mean of each metric, keyed '<metric>@<k>'.
    """
    return {
        f"{name}@{k}": float(torch.as_tensor(value).float().nanmean())
        for k, metrics in results.items()
        for name, value in metrics.items()
    }


@pytest.mark.parametrize("model_name", ["BPR", "FM"])
@pytest.mark.parametrize("strategy", ["full", "sampled"])
def test_estimate_times_every_evaluation_layout(
    dataset: Dataset, model_name: str, strategy: str
):
    model = make_model(model_name, dataset)
    config = _config()

    def loader():
        return retrieve_evaluation_dataloader(
            dataset=dataset,
            model=model,
            strategy=strategy,
            num_negatives=config.num_negatives,
        )

    measured = _estimate_eval_loop(
        evaluator=build_evaluator(config, dataset),
        model=model,
        dataloader=loader(),
        strategy=strategy,
        dataset=dataset,
        device="cpu",
        warmup_batches=1,
        measured_batches=1_000,
        tracker=EstimateStageTracker(baseline_rss_mb=0.0, device="cpu"),
    )

    assert measured["measured_batches"] == len(loader()) - 1
    assert all(t > 0 for t in measured["batch_times"])

    reference = build_evaluator(config, dataset)
    reference.evaluate(
        model=model, dataloader=loader(), strategy=strategy, dataset=dataset
    )
    expected = _as_floats(reference.compute_results())
    assert _as_floats(measured["results"]) == pytest.approx(expected)
