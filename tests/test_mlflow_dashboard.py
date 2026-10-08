"""What the MLflow dashboard sends to MLflow.

Every WarpRec metric is named ``<metric>@<k>``, and MLflow refuses metric names
holding an ``@``, so the dashboard used to fail on the first report. What MLflow
receives is spelled ``<metric>/<k>``; the names WarpRec uses everywhere else
are left alone.
"""

from typing import Any, Dict

import pytest

pytest.importorskip("mlflow")

from mlflow.tracking import MlflowClient  # noqa: E402

from warprec.recommenders.trainer.dashboard_callbacks import (  # noqa: E402
    WarpRecMLflowLoggerCallback,
    mlflow_metric_name,
)


class _Trial:
    """The part of a Ray Tune trial the MLflow callback reads."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.local_path = None

    def __str__(self) -> str:
        return "BPR_00000"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("nDCG@10", "nDCG/10"),
        ("Recall@20", "Recall/20"),
        ("loss", "loss"),
        ("ram_peak_mb", "ram_peak_mb"),
    ],
)
def test_metric_names_are_spelled_the_way_mlflow_accepts(name: str, expected: str):
    assert mlflow_metric_name(name) == expected


def test_a_report_reaches_a_real_mlflow_store(tmp_path):
    tracking_uri = (tmp_path / "mlruns").as_uri()
    callback = WarpRecMLflowLoggerCallback(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
        experiment_name="warprec-test",
    )
    callback.setup()
    trial = _Trial({"embedding_size": 8, "validation_metric_name": "nDCG@10"})
    result = {"nDCG@10": 0.25, "Recall@20": 0.5, "loss": 1.5, "training_iteration": 1}

    callback.log_trial_start(trial)
    callback.log_trial_result(1, trial, result)
    callback.log_trial_end(trial)

    run_id = callback._trial_runs[trial]  # pylint: disable=protected-access
    metrics = MlflowClient(tracking_uri=tracking_uri).get_run(run_id).data.metrics
    assert metrics["nDCG/10"] == 0.25
    assert metrics["Recall/20"] == 0.5
    assert metrics["loss"] == 1.5
    assert not any("@" in name for name in metrics)
    # The report itself is not rewritten for the rest of Ray Tune.
    assert "nDCG@10" in result
