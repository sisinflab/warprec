"""Behavioural tests for the pause and resume manifest.

Resuming a run is only safe if the manifest survives a round trip unchanged,
if a fingerprint notices the configuration changes that matter and ignores the
ones that do not, and if a manifest that cannot be trusted is refused rather
than half-read.
"""

import json
from pathlib import Path

import pytest
import torch

from warprec.common.run_state import (
    ModelState,
    ModelStatus,
    RunState,
    RunStateStore,
    model_fingerprint,
)
from warprec.data.writer import LocalWriter


def a_state() -> RunState:
    """A manifest with one model of every interesting status.

    Returns:
        RunState: The state under test.
    """
    state = RunState(
        run_name="run-1",
        pipeline="train",
        warprec_version="1.7.0",
        writer_timestamp="20260921",
        config_fingerprint="abc123",
    )
    state.models["ItemKNN"] = ModelState(
        status=ModelStatus.COMPLETED,
        fingerprint="f1",
        best_params={"k": 100, "similarity": "cosine"},
        best_iter=7,
        ray_report={"nDCG@10": 0.3},
        timing={"train": 1.5},
    )
    state.models["BPRMF"] = ModelState(
        status=ModelStatus.INTERRUPTED,
        fingerprint="f2",
        tune_experiment_name="bprmf_exp",
    )
    return state


@pytest.fixture()
def store(tmp_path: Path) -> RunStateStore:
    """A store backed by a real experiment directory.

    Args:
        tmp_path (Path): The temporary directory pytest provides.

    Returns:
        RunStateStore: The store under test.
    """
    writer = LocalWriter(dataset_name="ds", local_path=str(tmp_path))
    return RunStateStore(writer, "run-1")


def test_a_manifest_survives_a_round_trip():
    """Serialising and parsing a manifest preserves every field."""
    original = a_state()
    restored = RunState.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored.to_dict() == original.to_dict()
    assert restored.models["ItemKNN"].status is ModelStatus.COMPLETED
    assert restored.models["ItemKNN"].best_params == {"k": 100, "similarity": "cosine"}
    assert restored.models["BPRMF"].tune_experiment_name == "bprmf_exp"


def test_a_saved_manifest_reloads(store: RunStateStore):
    """A manifest written through the writer reads back the same."""
    original = a_state()
    store.save(original)

    reloaded = store.load()
    assert reloaded is not None
    assert reloaded.run_name == original.run_name
    assert set(reloaded.models) == set(original.models)
    assert reloaded.models["ItemKNN"].best_iter == 7


def test_no_manifest_reads_as_no_previous_run(store: RunStateStore):
    """An absent manifest is not an error; it means the run is new."""
    assert store.load() is None


def test_an_unreadable_manifest_is_refused(store: RunStateStore):
    """Corrupt JSON is treated as no previous run rather than half-parsed."""
    Path(store.state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(store.state_path).write_text("{not json", encoding="utf-8")

    assert store.load() is None


def test_a_manifest_from_a_newer_layout_is_rejected(store: RunStateStore):
    """A newer schema raises rather than resuming into an unknown layout."""
    payload = a_state().to_dict()
    payload["schema_version"] = 99_999
    Path(store.state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(store.state_path).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        store.load()


def test_asking_for_an_unknown_model_creates_a_pending_entry():
    """A model absent from the manifest starts pending rather than raising."""
    state = a_state()

    assert state.model_state("BrandNew").status is ModelStatus.PENDING
    assert "BrandNew" in state.models


def test_a_fingerprint_follows_the_parameters_that_matter():
    """Changing a hyperparameter invalidates a resume; the same block does not."""
    params = {"k": [50, 100], "optimization": {"num_samples": 10}}

    assert model_fingerprint("ItemKNN", params) == model_fingerprint(
        "ItemKNN", dict(params)
    )
    assert model_fingerprint("ItemKNN", params) != model_fingerprint(
        "ItemKNN", {"k": [50, 200], "optimization": {"num_samples": 10}}
    )
    assert model_fingerprint("ItemKNN", params) != model_fingerprint("BPRMF", params)


def test_a_fingerprint_ignores_the_resources_a_run_happened_to_use():
    """A paused run may be resumed on a differently sized cluster."""
    base = {"k": 100, "optimization": {"num_samples": 10, "device": "cpu"}}
    moved = {
        "k": 100,
        "optimization": {
            "num_samples": 10,
            "device": "cuda",
            "cpu_per_trial": 8,
            "gpu_per_trial": 2,
            "max_concurrent_trials": 16,
        },
    }

    assert model_fingerprint("ItemKNN", base) == model_fingerprint("ItemKNN", moved)


def test_evaluation_results_round_trip_through_the_store(store: RunStateStore):
    """The per-user tensors the significance tests need survive a resume."""
    results = {("nDCG", 10): torch.arange(5, dtype=torch.float32)}
    store.save_eval_results("ItemKNN", results)

    back = store.load_eval_results("ItemKNN")
    assert back is not None
    assert set(back) == set(results)
    torch.testing.assert_close(back[("nDCG", 10)], results[("nDCG", 10)])


def test_missing_evaluation_results_read_as_absent(store: RunStateStore):
    """A model whose results were never written reads back as None."""
    assert store.load_eval_results("NeverRan") is None
