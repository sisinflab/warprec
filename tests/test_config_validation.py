"""A wrong configuration fails when it is loaded, naming the key at fault.

Each test builds a configuration with one mistake and checks that validating it
raises, with a message that says which key is wrong, rather than the pipeline
failing later, deep inside the code that reads the value.
"""

from pathlib import Path
from typing import Any, Dict, List, Type

import pandas as pd
import pytest
from pydantic import ValidationError

from warprec.utils.config import config as config_module
from warprec.utils.config.splitter_configuration import SplittingConfig
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


NEUMF = {
    "mf_embedding_size": 8,
    "mlp_embedding_size": 8,
    "mlp_hidden_size": [16, 8],
    "mf_train": True,
    "mlp_train": True,
    "dropout": 0.1,
    "reg_weight": 0.001,
    "weight_decay": 0.0,
    "batch_size": 32,
    "epochs": 1,
    "learning_rate": 0.001,
    "neg_samples": 1,
}


@pytest.mark.parametrize(
    "models, key",
    [
        ({"EASE": {"l2": [50, 200]}}, "models.EASE.l2"),
        ({"EASE": {"l2": [50]}}, "models.EASE.l2"),
        (
            {
                "EASE": {
                    "l2": ["uniform", 10, 100],
                    "optimization": {"strategy": "random"},
                }
            },
            "models.EASE.l2",
        ),
        (
            {"NeuMF": {**NEUMF, "mlp_hidden_size": [[16, 8], [32]]}},
            "models.NeuMF.mlp_hidden_size",
        ),
        ({"NeuMF": {**NEUMF, "dropout": [0.1, 0.2]}}, "models.NeuMF.dropout"),
    ],
)
def test_the_design_pipeline_refuses_a_search_space(
    ratings: Path, models: Dict[str, Any], key: str
):
    """A design configuration takes single values, and says which key is not one."""
    with pytest.raises(ValidationError, match=key):
        load(DesignConfiguration, sections(ratings, models))


def test_the_design_pipeline_takes_a_list_valued_hyperparameter(ratings: Path):
    """A hyperparameter whose single value is a list, as layer sizes are, is accepted."""
    config = load(DesignConfiguration, sections(ratings, {"NeuMF": NEUMF}))

    assert config.models["NeuMF"]["mlp_hidden_size"] == [16, 8]


@pytest.mark.parametrize("pipeline", ALL_PIPELINES)
def test_an_unknown_hyperparameter_is_refused_naming_it(
    pipeline: Type[WarpRecConfiguration], ratings: Path
):
    """A key the model does not take fails at load, listing the ones it does."""
    config = sections(ratings, {"FM": {**FM, "weight_decay": 0.01}})
    config["reader"]["labels"] = {"context_labels": ["timestamp"]}

    with pytest.raises(ValidationError, match="'weight_decay'") as error:
        load(pipeline, config)

    assert "reg_weight" in str(error.value)


def attentions(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Record what the configuration logs as needing the user's attention.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the logger's method.

    Returns:
        List[str]: The messages, filled as they are logged.
    """
    messages: List[str] = []
    monkeypatch.setattr(
        config_module.logger,
        "attention",
        lambda message, *a, **k: messages.append(message),
    )
    return messages


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_filtering_pre_split_data_is_warned_about(
    pipeline: Type[WarpRecConfiguration],
    ratings: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Filtering is not applied to split data, and loading says so, without failing."""
    split = ratings.parent / "split"
    split.mkdir()
    (split / "train.tsv").write_text(ratings.read_text())
    messages = attentions(monkeypatch)

    load(
        pipeline,
        sections(
            ratings,
            {"Pop": {}},
            reader={
                "loading_strategy": "split",
                "data_type": "transaction",
                "reading_method": "local",
                "rating_type": "implicit",
                "split": {"local_path": str(split)},
            },
            filtering={"UserMin": {"min_interactions": 2}},
        ),
    )

    assert any("'filtering'" in m and "split" in m for m in messages)


def test_filtering_a_dataset_is_not_warned_about(
    ratings: Path, monkeypatch: pytest.MonkeyPatch
):
    """The warning is only for split data, which the filters never see."""
    messages = attentions(monkeypatch)

    load(
        DesignConfiguration,
        sections(ratings, {"Pop": {}}, filtering={"UserMin": {"min_interactions": 2}}),
    )

    assert not any("'filtering'" in m for m in messages)


@pytest.mark.parametrize("strategy", ["item_cold_start", "user_cold_start"])
def test_a_cold_start_split_requires_its_ratio(strategy: str):
    """The cold-start strategies read a ratio, so the configuration requires one."""
    with pytest.raises(ValidationError, match="'ratio'"):
        SplittingConfig(test_splitting={"strategy": strategy})


@pytest.mark.parametrize(
    "test, validation",
    [
        ("random_holdout", "item_cold_start"),
        ("temporal_holdout", "user_cold_start"),
        ("user_cold_start", "item_cold_start"),
    ],
)
def test_a_cold_start_validation_split_that_would_be_emptied_is_refused(
    test: str, validation: str
):
    """A validation split holding out what the test split keeps is refused at load."""
    with pytest.raises(ValidationError, match="validation_splitting"):
        SplittingConfig(
            test_splitting={"strategy": test, "ratio": 0.1},
            validation_splitting={"strategy": validation, "ratio": 0.1},
        )


def test_a_cold_start_validation_split_matching_the_test_split_is_accepted():
    """Validating on held-out items when the test holds out items too is a protocol."""
    SplittingConfig(
        test_splitting={"strategy": "item_cold_start", "ratio": 0.1},
        validation_splitting={"strategy": "item_cold_start", "ratio": 0.1},
    )
