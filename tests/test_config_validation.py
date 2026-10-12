"""A wrong configuration fails when it is loaded, naming the key at fault.

Each test builds a configuration with one mistake and checks that validating it
raises, with a message that says which key is wrong, rather than the pipeline
failing later, deep inside the code that reads the value.
"""

from pathlib import Path
from typing import Any, Dict, Type

import pandas as pd
import pytest
from pydantic import ValidationError

from warprec.utils.config.config import (
    DesignConfiguration,
    EstimateConfiguration,
    EvalConfiguration,
    TrainConfiguration,
    WarpRecConfiguration,
)

PIPELINES = [TrainConfiguration, DesignConfiguration, EvalConfiguration]
ALL_PIPELINES = PIPELINES + [EstimateConfiguration]

CKE = {
    "embedding_size": 8,
    "kg_embedding_size": 8,
    "reg_weight": 0.01,
    "kg_reg_weight": 0.01,
    "batch_size": 32,
    "epochs": 1,
    "learning_rate": 0.01,
}
VBPR = {
    "embedding_size": 8,
    "reg_weight": 0.01,
    "batch_size": 32,
    "epochs": 1,
    "learning_rate": 0.01,
}
FM = {
    "embedding_size": 8,
    "reg_weight": 0.01,
    "batch_size": 32,
    "epochs": 1,
    "learning_rate": 0.01,
    "neg_samples": 1,
}


@pytest.fixture
def ratings(tmp_path: Path) -> Path:
    """A small ratings file for the reader section to point at.

    Args:
        tmp_path (Path): Where the file is written.

    Returns:
        Path: The file.
    """
    path = tmp_path / "ratings.tsv"
    pd.DataFrame(
        {"user_id": [0, 0, 1, 1], "item_id": [0, 1, 0, 2], "rating": 1.0}
    ).assign(timestamp=range(4)).to_csv(path, sep="\t", index=False)
    return path


def sections(ratings: Path, models: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    """The sections of a valid configuration, with the given models.

    Args:
        ratings (Path): The ratings file.
        models (Dict[str, Any]): The models section.
        **overrides (Any): Sections replacing or adding to the defaults.

    Returns:
        Dict[str, Any]: The configuration, ready to be validated.
    """
    config: Dict[str, Any] = {
        "reader": {
            "loading_strategy": "dataset",
            "data_type": "transaction",
            "reading_method": "local",
            "local_path": str(ratings),
            "rating_type": "implicit",
        },
        "writer": {
            "dataset_name": "validation",
            "writing_method": "local",
            "local_experiment_path": str(ratings.parent / "out"),
        },
        "splitter": {"test_splitting": {"strategy": "temporal_holdout", "ratio": 0.2}},
        "models": models,
        "evaluation": {"top_k": [5], "metrics": ["nDCG"]},
    }
    config.update(overrides)
    return config


def load(
    pipeline: Type[WarpRecConfiguration], config: Dict[str, Any]
) -> WarpRecConfiguration:
    """Validate a configuration as the given pipeline loads it.

    Args:
        pipeline (Type[WarpRecConfiguration]): The pipeline's configuration class.
        config (Dict[str, Any]): The configuration.

    Returns:
        WarpRecConfiguration: The validated configuration.
    """
    if pipeline is DesignConfiguration:
        config = {k: v for k, v in config.items() if k != "writer"}
    return pipeline(**config)


@pytest.mark.parametrize("pipeline", ALL_PIPELINES)
@pytest.mark.parametrize(
    "models, message",
    [
        ({"CKE": CKE}, "knowledge graph"),
        ({"VBPR": VBPR}, "multimodal item features"),
        ({"AddEASE": {"l2": 10.0, "alpha": 0.5}}, "requires side information"),
        ({"FM": FM}, "context_labels"),
        ({"NotAModel": {}}, "'NotAModel' is not registered"),
    ],
)
def test_a_model_missing_its_input_is_refused_by_every_pipeline(
    pipeline: Type[WarpRecConfiguration],
    ratings: Path,
    models: Dict[str, Any],
    message: str,
):
    """Every pipeline checks at load that each model has what it reads."""
    with pytest.raises(ValidationError, match=message):
        load(pipeline, sections(ratings, models))


@pytest.mark.parametrize("pipeline", ALL_PIPELINES)
def test_a_model_naming_an_unconfigured_modality_is_refused_by_every_pipeline(
    pipeline: Type[WarpRecConfiguration], ratings: Path
):
    """Every pipeline checks at load that a model's modalities are configured."""
    config = sections(ratings, {"VBPR": {**VBPR, "modalities": ["audio"]}})
    config["reader"]["multimodal"] = {
        "visual": {"local_path": "visual.npy", "item_path": "items.tsv"}
    }
    with pytest.raises(ValidationError, match=r"modalities \['audio'\]"):
        load(pipeline, config)


@pytest.mark.parametrize(
    "pipeline", [DesignConfiguration, EvalConfiguration, EstimateConfiguration]
)
def test_single_values_stay_single_values(
    pipeline: Type[WarpRecConfiguration], ratings: Path
):
    """Validating the models does not turn what the model is built from into spaces."""
    config = load(pipeline, sections(ratings, {"EASE": {"l2": 10.0}}))

    assert config.models == {"EASE": {"l2": 10.0}}
