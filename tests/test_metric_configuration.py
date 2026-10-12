"""The metric names of evaluation.metrics are checked when the configuration loads.

An unknown name used to pass validation and only fail when the evaluator was
built, after the data had been read and split.
"""

import uuid
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from warprec.utils.config.config import load_design_configuration
from warprec.utils.config.evaluation_configuration import EvaluationConfig

METRIC_SOURCE = """
from warprec.evaluation.metrics.accuracy.precision import Precision
from warprec.utils.registry import metric_registry


@metric_registry.register("{name}")
class {name}(Precision):
    pass
"""


def test_an_unknown_metric_fails_at_load_naming_it():
    """The message names the key and the metric, and offers the close match."""
    with pytest.raises(ValidationError) as error:
        EvaluationConfig(top_k=[5], metrics=["nDCG", "Recal"])

    message = str(error.value)
    assert "evaluation.metrics" in message
    assert "'Recal'" in message
    assert "Did you mean 'Recall'" in message


def test_an_unknown_complex_metric_offers_the_close_match():
    """complex_metrics was already checked; it now says what was meant too."""
    with pytest.raises(ValidationError, match="Did you mean 'Precision'"):
        EvaluationConfig(
            top_k=[5], metrics=["nDCG"], complex_metrics=[{"name": "Precison"}]
        )


def test_metric_names_are_matched_case_insensitively():
    """The registry ignores case, so the check does too."""
    config = EvaluationConfig(top_k=[5], metrics=["ndcg", "RECALL", "HitRate"])

    assert config.metrics == ["ndcg", "RECALL", "HitRate"]


def test_a_name_with_no_close_match_lists_where_to_look():
    """Without a suggestion the message still says what to do."""
    with pytest.raises(ValidationError, match="registered metric"):
        EvaluationConfig(top_k=[5], metrics=["Zzzzzz"])


def test_a_custom_metric_can_be_listed_by_name(tmp_path: Path):
    """Custom modules load before validation, so their metrics are known."""
    name = f"Metric{uuid.uuid4().hex[:8]}"
    module = tmp_path / "user_code" / f"{name.lower()}.py"
    module.parent.mkdir(parents=True)
    module.write_text(METRIC_SOURCE.format(name=name))
    data = tmp_path / "data.tsv"
    pd.DataFrame(
        {"user_id": [1, 1, 2, 2], "item_id": [1, 2, 1, 3], "timestamp": [1, 2, 3, 4]}
    ).to_csv(data, sep="\t", index=False)
    config = tmp_path / "config.yml"
    config.write_text(
        yaml.safe_dump(
            {
                "reader": {
                    "loading_strategy": "dataset",
                    "data_type": "transaction",
                    "reading_method": "local",
                    "local_path": str(data),
                    "rating_type": "implicit",
                },
                "splitter": {
                    "test_splitting": {"strategy": "temporal_holdout", "ratio": 0.5}
                },
                "models": {"Pop": {}},
                "evaluation": {"top_k": [2], "metrics": ["nDCG", name]},
                "general": {"custom_modules": [str(module)]},
            }
        )
    )

    loaded = load_design_configuration(str(config))

    assert loaded.evaluation.metrics == ["nDCG", name]
