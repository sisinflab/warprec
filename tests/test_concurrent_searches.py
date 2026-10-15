"""Searches running at the same time each run their own model.

The swarm optimises every model at once, each from a Ray task of its own.
Ray Tune registers the trainable of a search in the cluster under its name,
for the job: when every search registered a function of the same name, the
last one overwrote the others, and a trial created afterwards ran another
model's search with its own hyperparameters (BPR's trials ran as EASE).
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
import ray
from ray.tune import RunConfig, Tuner

import warprec.recommenders  # noqa: F401  (populates the registries)
from warprec.data.dataset import Dataset
from warprec.recommenders.trainer.trainer import Trainer
from warprec.utils.config.evaluation_configuration import EvaluationConfig
from warprec.utils.registry import params_registry

SEARCHES: Dict[str, Dict[str, Any]] = {
    "BPR": {
        "embedding_size": ["grid", 8, 16],
        "reg_weight": 1e-5,
        "batch_size": 256,
        "epochs": 1,
        "learning_rate": 0.001,
    },
    "EASE": {"l2": ["grid", 10.0, 50.0]},
}
NEEDED_CPUS = 3


@pytest.fixture(scope="module")
def local_ray() -> Iterator[None]:
    """A local Ray instance with room for one trial of each search at once.

    Yields:
        None: While the tests run.
    """
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=4, include_dashboard=False)
    elif ray.cluster_resources().get("CPU", 0) < NEEDED_CPUS:
        pytest.skip("The running Ray instance has too few CPUs for two searches.")
    yield
    if started:
        ray.shutdown()


@ray.remote(num_cpus=0)
def search(
    model_name: str, dataset: Dataset, storage_path: str
) -> List[Dict[str, Any]]:
    """Run one model's search, as a swarm's model pipeline does.

    Args:
        model_name (str): The model to optimise.
        dataset (Dataset): The dataset to optimise on.
        storage_path (str): Where Ray Tune stores its results.

    Returns:
        List[Dict[str, Any]]: For every trial, its hyperparameters, the keys
            of its last report and its error, if any.
    """
    # pylint: disable = import-outside-toplevel, unused-import, reimported
    import warprec.recommenders  # noqa: F401,F811  (the task's own registries)

    params = params_registry.get(
        model_name,
        **SEARCHES[model_name],
        optimization={"num_workers": 0, "max_concurrent_trials": 1},
    )
    trainer = Trainer(storage_path=storage_path, run_name="concurrent")
    results = trainer._setup_tuner(
        model_name=model_name,
        params=params,
        dataset=dataset,
        evaluation=EvaluationConfig(
            top_k=[5], metrics=["nDCG"], validation_metric="nDCG@5"
        ),
        validation_score="nDCG@5",
        device="cpu",
        ray_verbose=0,
    ).fit()
    return [
        {
            "config": dict(result.config),
            "keys": sorted(result.metrics or {}),
            "error": None if result.error is None else str(result.error)[:300],
        }
        for result in results  # type: ignore[attr-defined]
    ]


def test_concurrent_searches_each_run_their_own_model(
    local_ray: None, dataset: Dataset, tmp_path: Path
):
    """Every trial of BPR trains BPR, every trial of EASE fits EASE."""
    reference = ray.put(dataset)
    pending = {
        model_name: search.remote(model_name, reference, str(tmp_path))
        for model_name in SEARCHES
    }
    trials = {model_name: ray.get(task) for model_name, task in pending.items()}

    for model_name, model_trials in trials.items():
        assert len(model_trials) == 2, model_name
        for trial in model_trials:
            assert trial["error"] is None, (model_name, trial)
            # Only the iterative model reports a training loss and epochs
            trained_epochs = "train_loss" in trial["keys"]
            assert trained_epochs == (model_name == "BPR"), (model_name, trial)
            assert "nDCG@5" in trial["keys"], (model_name, trial)

    assert sorted(t["config"]["embedding_size"] for t in trials["BPR"]) == [8, 16]
    assert sorted(t["config"]["l2"] for t in trials["EASE"]) == [10.0, 50.0]


def _bpr_params() -> Any:
    """BPR's configuration in the searches below.

    Returns:
        Any: The model configuration.
    """
    return params_registry.get(
        "BPR",
        **SEARCHES["BPR"],
        optimization={"num_workers": 0, "max_concurrent_trials": 1},
    )


def _bpr_search(trainer: Trainer, dataset: Dataset) -> Tuner:
    """Prepare BPR's search, as the train pipeline does, without running it.

    Args:
        trainer (Trainer): The trainer of the run.
        dataset (Dataset): The dataset to optimise on.

    Returns:
        Tuner: The search, new or restored.
    """
    return trainer._setup_tuner(
        model_name="BPR",
        params=_bpr_params(),
        dataset=dataset,
        evaluation=EvaluationConfig(
            top_k=[5], metrics=["nDCG"], validation_metric="nDCG@5"
        ),
        validation_score="nDCG@5",
        device="cpu",
        ray_verbose=0,
    )


def test_a_resumed_search_finds_its_trainable_under_its_name(
    local_ray: None, dataset: Dataset, tmp_path: Path
):
    """The name of a search's trainable is the same when its run resumes."""
    _bpr_search(Trainer(storage_path=str(tmp_path), run_name="run"), dataset)
    stored = Trainer._stored_trainable_name(str(tmp_path / "run__BPR"))

    assert stored is not None and stored.startswith("warprec__run__BPR__")
    _bpr_search(
        Trainer(storage_path=str(tmp_path), run_name="run", resumable=True), dataset
    )
    assert Trainer._stored_trainable_name(str(tmp_path / "run__BPR")) == stored


def test_a_search_paused_by_an_older_version_still_resumes(
    local_ray: None, dataset: Dataset, tmp_path: Path
):
    """A search whose trainable was registered as 'train_driver_fn' restores."""

    def train_driver_fn(config: dict) -> None:
        """The trainable of a search started by WarpRec 2.2.2.

        Args:
            config (dict): The trial's hyperparameters.
        """

    Tuner(
        train_driver_fn,
        param_space=Trainer(storage_path="unused")._parse_params(_bpr_params()),
        run_config=RunConfig(name="run__BPR", storage_path=str(tmp_path)),
    )

    restored = _bpr_search(
        Trainer(storage_path=str(tmp_path), run_name="run", resumable=True), dataset
    )

    assert restored is not None
