"""A key no configuration section reads is reported when the file is loaded.

Such a key is ignored, so a misspelled one would otherwise leave the run on the
default without a word. Loading logs one line per key, with its dotted path and
the closest key the section does read, and goes on: the file still loads.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd
import pytest
import yaml

from warprec.errors import ConfigurationError
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.config import common as common_module
from warprec.utils.config.config import (
    load_design_configuration,
    load_estimate_configuration,
    load_eval_configuration,
    load_train_configuration,
)
from warprec.utils.registry import model_registry

BPR: Dict[str, Any] = {
    "embedding_size": 8,
    "reg_weight": 0.0,
    "batch_size": 8,
    "epochs": 1,
    "learning_rate": 0.01,
}


def attentions(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Record what loading logs as needing the user's attention.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the logger's method.

    Returns:
        List[str]: The messages, filled as they are logged.
    """
    messages: List[str] = []
    monkeypatch.setattr(
        common_module.logger,
        "attention",
        lambda message, *a, **k: messages.append(message),
    )
    return messages


def write(tmp_path: Path, frame: pd.DataFrame, **sections: Any) -> Path:
    """A configuration over the shared interactions, written to disk.

    Args:
        tmp_path (Path): Where to write the data and the configuration.
        frame (pd.DataFrame): The interactions to read.
        **sections (Any): Top-level sections replacing or adding to the defaults.

    Returns:
        Path: The configuration's path.
    """
    data = tmp_path / "data.tsv"
    frame[["user_id", "item_id", "rating", "timestamp"]].to_csv(
        data, sep="\t", index=False
    )
    config: Dict[str, Any] = {
        "reader": {
            "loading_strategy": "dataset",
            "data_type": "transaction",
            "reading_method": "local",
            "local_path": str(data),
            "rating_type": "implicit",
        },
        "splitter": {"test_splitting": {"strategy": "temporal_holdout", "ratio": 0.2}},
        "models": {"BPR": dict(BPR)},
        "evaluation": {"top_k": [5], "metrics": ["nDCG"]},
    }
    config.update(sections)
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    return path


def writer(tmp_path: Path) -> Dict[str, Any]:
    """A writer section for the pipelines that read one.

    Args:
        tmp_path (Path): Where the experiment would be written.

    Returns:
        Dict[str, Any]: The section.
    """
    return {
        "dataset_name": "keys",
        "writing_method": "local",
        "local_experiment_path": str(tmp_path / "out"),
    }


def test_a_valid_configuration_logs_nothing(
    tmp_path: Path, interactions_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    load_design_configuration(str(write(tmp_path, interactions_frame)))
    assert messages == []


def test_a_misspelled_section_is_named_with_the_section_meant(
    tmp_path: Path, interactions_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    path = write(
        tmp_path, interactions_frame, evalution={"top_k": [10], "metrics": ["Recall"]}
    )

    config = load_design_configuration(str(path))

    assert config.evaluation.top_k == [5]
    assert len(messages) == 1
    assert "'evalution'" in messages[0]
    assert messages[0].endswith("Did you mean 'evaluation'?")


def test_a_misspelled_key_is_named_by_its_dotted_path(
    tmp_path: Path, interactions_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    path = write(
        tmp_path,
        interactions_frame,
        evaluation={"top_k": [5], "top_kk": [10], "metrics": ["nDCG"]},
    )

    load_design_configuration(str(path))

    assert len(messages) == 1
    assert "'evaluation.top_kk'" in messages[0]
    assert messages[0].endswith("Did you mean 'evaluation.top_k'?")


def test_a_key_close_to_nothing_gets_no_suggestion(
    tmp_path: Path, interactions_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    path = write(
        tmp_path,
        interactions_frame,
        general={"device": "cpu", "zzzz": 1},
    )

    load_design_configuration(str(path))

    assert len(messages) == 1
    assert "'general.zzzz'" in messages[0]
    assert "Did you mean" not in messages[0]


@pytest.mark.parametrize(
    "loader",
    [
        load_design_configuration,
        load_eval_configuration,
        load_estimate_configuration,
        load_train_configuration,
    ],
)
def test_sections_other_pipelines_read_are_not_reported(
    loader: Callable[[str], Any],
    tmp_path: Path,
    interactions_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
):
    """One file may serve every pipeline, each ignoring the others' sections."""
    messages = attentions(monkeypatch)
    path = write(
        tmp_path,
        interactions_frame,
        writer=writer(tmp_path),
        dashboard={"wandb": {"enabled": False}},
        run={"resume": "never"},
        estimate={"train_batches": 5},
    )

    loader(str(path))

    assert messages == []


def test_a_typo_inside_a_section_another_pipeline_reads_is_named(
    tmp_path: Path, interactions_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    """The design pipeline does not read 'writer', but no pipeline reads 'sepp'."""
    messages = attentions(monkeypatch)
    path = write(
        tmp_path,
        interactions_frame,
        writer={**writer(tmp_path), "results": {"sepp": ","}},
    )

    load_design_configuration(str(path))

    assert len(messages) == 1
    assert messages[0].endswith("Did you mean 'writer.results.sep'?")


def test_a_misspelled_section_of_another_pipeline_gets_its_name(
    tmp_path: Path, interactions_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    path = write(tmp_path, interactions_frame, estimat={"train_batches": 5})

    load_design_configuration(str(path))

    assert len(messages) == 1
    assert messages[0].endswith("Did you mean 'estimate'?")


def nested_typos(tmp_path: Path) -> Dict[str, Any]:
    """Sections each holding one misspelled key, deep in the configuration.

    Args:
        tmp_path (Path): Where the experiment would be written.

    Returns:
        Dict[str, Any]: The sections, to write over the defaults.
    """
    return {
        "splitter": {
            "test_splitting": {"strategy": "temporal_holdout", "ratio": 0.2, "sede": 1},
        },
        "models": {
            "BPR": {
                **BPR,
                "meta": {"save_modle": True},
                "optimization": {"properties": {"seeed": 3}},
                "early_stopping": {"patience": 2, "grace_periodd": 1},
            }
        },
        "writer": {**writer(tmp_path), "results": {"sepp": ","}},
        "dashboard": {"wandb": {"projcet": "x"}},
        "run": {"resum": "never"},
    }


NESTED = [
    ("splitter.test_splitting.sede", "splitter.test_splitting.seed"),
    ("models.BPR.meta.save_modle", "models.BPR.meta.save_model"),
    (
        "models.BPR.optimization.properties.seeed",
        "models.BPR.optimization.properties.seed",
    ),
    (
        "models.BPR.early_stopping.grace_periodd",
        "models.BPR.early_stopping.grace_period",
    ),
    ("writer.results.sepp", "writer.results.sep"),
    ("dashboard.wandb.projcet", "dashboard.wandb.project"),
    ("run.resum", "run.resume"),
]


def test_misspelled_keys_in_nested_sections_are_each_named(
    tmp_path: Path, interactions_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    path = write(tmp_path, interactions_frame, **nested_typos(tmp_path))

    load_train_configuration(str(path))

    assert len(messages) == len(NESTED)
    for key, meant in NESTED:
        (message,) = [m for m in messages if f"'{key}'" in m]
        assert message.endswith(f"Did you mean '{meant}'?")


@pytest.mark.parametrize(
    "loader", [load_eval_configuration, load_estimate_configuration]
)
def test_every_pipeline_names_unknown_keys(
    loader: Callable[[str], Any],
    tmp_path: Path,
    interactions_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
):
    messages = attentions(monkeypatch)
    path = write(
        tmp_path,
        interactions_frame,
        writer=writer(tmp_path),
        reader={
            "loading_strategy": "dataset",
            "data_type": "transaction",
            "reading_method": "local",
            "local_path": str(tmp_path / "data.tsv"),
            "rating_type": "implicit",
            "labels": {"user_id_lable": "user_id"},
        },
    )

    loader(str(path))

    assert len(messages) == 1
    assert messages[0].endswith("Did you mean 'reader.labels.user_id_label'?")


def test_an_unknown_estimate_key_is_named(
    tmp_path: Path, interactions_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    path = write(
        tmp_path,
        interactions_frame,
        writer=writer(tmp_path),
        estimate={"train_batchs": 5},
    )

    load_estimate_configuration(str(path))

    assert len(messages) == 1
    assert messages[0].endswith("Did you mean 'estimate.train_batches'?")


def test_free_form_entries_are_not_reported(
    tmp_path: Path, interactions_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    path = write(
        tmp_path,
        interactions_frame,
        writer=writer(tmp_path),
        models={
            "BPR": {
                **BPR,
                "optimization": {
                    "optimizer": {"name": "adam", "params": {"anything": 1}},
                    "lr_scheduler": {"name": "steplr", "params": {"step_size": 2}},
                },
            }
        },
        general={
            "callback": {"kwargs": {"anything": 1}},
            "custom_resources_data_prep": {"anything": 1},
        },
        evaluation={
            "top_k": [5],
            "metrics": ["nDCG"],
            "complex_metrics": [{"name": "EPC", "params": {"anything": 1}}],
        },
        rerank={"params": {"anything": 1}},
        dashboard={"mlflow": {"tags": {"anything": 1}}},
    )

    load_train_configuration(str(path))

    assert messages == []


def test_a_misspelled_key_in_a_list_entry_is_named(
    tmp_path: Path, interactions_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    path = write(
        tmp_path,
        interactions_frame,
        evaluation={
            "top_k": [5],
            "metrics": ["nDCG"],
            "complex_metrics": [{"name": "EPC", "parmas": {}}],
        },
    )

    load_design_configuration(str(path))

    assert len(messages) == 1
    assert messages[0].endswith("Did you mean 'evaluation.complex_metrics[0].params'?")


def test_a_typo_that_fails_validation_is_still_named(
    tmp_path: Path, interactions_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    messages = attentions(monkeypatch)
    path = write(tmp_path, interactions_frame)
    config = yaml.safe_load(path.read_text())
    config["evalution"] = config.pop("evaluation")
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ConfigurationError):
        load_design_configuration(str(path))

    assert len(messages) == 1
    assert messages[0].endswith("Did you mean 'evaluation'?")


class Unvalidated(Recommender):
    """A model registered without a parameter class.

    Attributes:
        factor (int): Its one hyperparameter.
    """

    factor: int


@pytest.mark.parametrize(
    "loader", [load_design_configuration, load_train_configuration]
)
def test_a_key_a_model_without_parameter_class_drops_is_named(
    loader: Callable[[str], Any],
    tmp_path: Path,
    interactions_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(model_registry._registry, "UNVALIDATED", Unvalidated)
    messages = attentions(monkeypatch)
    path = write(
        tmp_path,
        interactions_frame,
        writer=writer(tmp_path),
        models={"Unvalidated": {"factor": 2, "factr": 3}},
    )

    loader(str(path))

    unread = [m for m in messages if "models.Unvalidated" in m]
    assert len(unread) == 1
    assert "'models.Unvalidated.factr'" in unread[0]
    assert unread[0].endswith("Did you mean 'models.Unvalidated.factor'?")
