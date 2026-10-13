"""What the estimate report says about a small dataset and a grid of setups.

Three things it used to get wrong: the RAM peak of a model was the mean of the
peaks of its setups, so the largest setup's need was under-reported; a
closed-form model was never evaluated, so it had no evaluation time; and with
the default warm-up of ten batches a loader of ten batches or fewer was all
warm-up, so the report said 'Measured Eval Batches: 0' with nothing to warn.
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from warprec.pipelines.estimate import _aggregate_reports, estimate_pipeline


def _report(ram_max: float, ram_min: float, eval_time: float) -> Dict[str, Any]:
    """One setup's report, with only the fields under test filled in.

    Args:
        ram_max (float): The setup's RAM peak.
        ram_min (float): The setup's RAM minimum.
        eval_time (float): The setup's estimated evaluation time.

    Returns:
        Dict[str, Any]: The report.
    """
    return {
        "Measured Train Batches": 5,
        "Measured Eval Batches": 1,
        "Train RAM Min": ram_min,
        "Train RAM Avg": (ram_min + ram_max) / 2,
        "Train RAM Max": ram_max,
        "Estimated Evaluation Time": eval_time,
        "Notes": "CPU only",
    }


def test_the_peak_of_a_model_is_the_largest_peak_of_its_setups():
    """Max and Min aggregate as such; the other figures stay averages."""
    model = _aggregate_reports(
        "BPR", [_report(10.0, 2.0, 1.0), _report(30.0, 4.0, 3.0)]
    )

    assert model["Train RAM Max"] == 30.0
    assert model["Train RAM Min"] == 2.0
    assert model["Train RAM Avg"] == 11.5
    assert model["Estimated Evaluation Time"] == 2.0


def test_every_model_gets_measured_batches_and_an_evaluation_time(
    tmp_path: Path, interactions_frame: pd.DataFrame
):
    """A loader shorter than the warm-up still has a batch measured."""
    data = tmp_path / "data.tsv"
    interactions_frame[["user_id", "item_id", "rating", "timestamp"]].to_csv(
        data, sep="\t", index=False
    )
    config = {
        "reader": {
            "loading_strategy": "dataset",
            "data_type": "transaction",
            "reading_method": "local",
            "local_path": str(data),
            "rating_type": "implicit",
            "sep": "\t",
        },
        "writer": {
            "dataset_name": "estimate",
            "writing_method": "local",
            "local_experiment_path": str(tmp_path / "runs"),
        },
        "splitter": {"test_splitting": {"strategy": "temporal_holdout", "ratio": 0.2}},
        "models": {
            "EASE": {"l2": 10.0, "optimization": {"num_workers": 0}},
            "BPR": {
                "embedding_size": 8,
                "reg_weight": 1e-5,
                "batch_size": 128,
                "epochs": 2,
                "learning_rate": 0.001,
                "optimization": {"num_workers": 0},
            },
        },
        # The default evaluation batch makes one batch of the 40 users, and
        # the default warm-up is ten batches
        "evaluation": {"top_k": [5], "metrics": ["nDCG"]},
        "general": {"device": "cpu"},
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))

    estimate_pipeline(str(path))

    report_file = next((tmp_path / "runs").rglob("Estimate_Report_*.tsv"))
    report = pd.read_csv(report_file, sep="\t").set_index("Model Name")

    for model in ("EASE", "BPR"):
        assert report.loc[model, "Measured Eval Batches"] >= 1
        assert pd.notna(report.loc[model, "Estimated Evaluation Time"])
        assert "warm-up reduced" in report.loc[model, "Notes"]
    assert report.loc["BPR", "Measured Train Batches"] >= 1
    # The model's own note replaces the generic one instead of repeating it
    assert report.loc["EASE", "Notes"].lower().count("analytical train-space") == 1
