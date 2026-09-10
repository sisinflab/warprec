import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import torch

from warprec.utils.logger import logger

if TYPE_CHECKING:
    from warprec.data.writer import Writer
    from warprec.utils.config import TrainConfiguration

# Bumped whenever the on-disk manifest layout changes in a way that an older
# WarpRec installation could not read correctly.
SCHEMA_VERSION = 1

# Fields excluded from fingerprints: changing them must not prevent a resume,
# because a paused run is routinely resumed on a differently sized cluster.
_EXCLUDED_OPTIMIZATION_FIELDS = frozenset(
    {
        "cpu_per_trial",
        "gpu_per_trial",
        "custom_resources_per_trial",
        "label_selector",
        "max_concurrent_trials",
        "num_workers",
        "device",
    }
)


def warprec_version() -> str:
    """Returns the version of the installed WarpRec package.

    Returns:
        str: The version string, or 'unknown' when the package metadata is
            not available (for instance when running from a source checkout
            that has not been installed).
    """
    try:
        return version("warprec")
    except PackageNotFoundError:
        return "unknown"


class ModelStatus(str, Enum):
    """Represents how far a single model progressed within a run.

    This enum is used to track the possible per-model states:
        - PENDING: The model has not been started.
        - INTERRUPTED: Hyperparameter optimization started but was paused.
        - HPO_COMPLETED: Hyperparameter optimization finished, but the stages
            that follow it did not all complete.
        - COMPLETED: Every stage for this model finished and was written.
        - FAILED: Hyperparameter optimization produced no usable model.
    """

    PENDING = "pending"
    INTERRUPTED = "interrupted"
    HPO_COMPLETED = "hpo_completed"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ModelState:
    """The recorded progress of one model inside a run.

    Attributes:
        status (ModelStatus): How far this model progressed.
        fingerprint (str): Fingerprint of the model's configuration.
        tune_experiment_name (Optional[str]): Name of the Ray Tune experiment.
        best_params (Optional[Dict[str, Any]]): Best hyperparameters found.
        best_iter (int): Best training iteration.
        best_checkpoint_path (Optional[str]): Path of the best trial checkpoint.
        ray_report (Dict[str, Any]): Summary report produced by the trainer.
        timing (Dict[str, Any]): Timing measurements already collected.
    """

    status: ModelStatus = ModelStatus.PENDING
    fingerprint: str = ""
    tune_experiment_name: Optional[str] = None
    best_params: Optional[Dict[str, Any]] = None
    best_iter: int = 0
    best_checkpoint_path: Optional[str] = None
    ray_report: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the model state.

        Returns:
            Dict[str, Any]: A JSON-serializable representation.
        """
        return {
            "status": self.status.value,
            "fingerprint": self.fingerprint,
            "tune_experiment_name": self.tune_experiment_name,
            "best_params": self.best_params,
            "best_iter": self.best_iter,
            "best_checkpoint_path": self.best_checkpoint_path,
            "ray_report": self.ray_report,
            "timing": self.timing,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelState":
        """Deserializes a model state.

        Args:
            data (Dict[str, Any]): The serialized representation.

        Returns:
            ModelState: The reconstructed model state.
        """
        return cls(
            status=ModelStatus(data.get("status", ModelStatus.PENDING.value)),
            fingerprint=data.get("fingerprint", ""),
            tune_experiment_name=data.get("tune_experiment_name"),
            best_params=data.get("best_params"),
            best_iter=data.get("best_iter", 0),
            best_checkpoint_path=data.get("best_checkpoint_path"),
            ray_report=data.get("ray_report", {}),
            timing=data.get("timing", {}),
        )


@dataclass
class RunState:
    """The persisted progress of a whole run.

    Attributes:
        run_name (str): The identifier of the run.
        pipeline (str): The pipeline that produced this state, 'train' or 'swarm'.
        warprec_version (str): The WarpRec version that created the run.
        writer_timestamp (str): The timestamp pinned into the run's output files.
        config_fingerprint (str): Fingerprint of the run-level configuration.
        created_at (str): ISO timestamp of creation.
        updated_at (str): ISO timestamp of the last save.
        models (Dict[str, ModelState]): Per-model progress, keyed by model name.
        schema_version (int): Layout version of the manifest.
    """

    run_name: str
    pipeline: str
    warprec_version: str
    writer_timestamp: str
    config_fingerprint: str
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    models: Dict[str, ModelState] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def model_state(self, model_name: str) -> ModelState:
        """Returns the state of a model, creating a pending entry when absent.

        Args:
            model_name (str): The name of the model.

        Returns:
            ModelState: The state of the requested model.
        """
        if model_name not in self.models:
            self.models[model_name] = ModelState()
        return self.models[model_name]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the run state.

        Returns:
            Dict[str, Any]: A JSON-serializable representation.
        """
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "pipeline": self.pipeline,
            "warprec_version": self.warprec_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "writer_timestamp": self.writer_timestamp,
            "config_fingerprint": self.config_fingerprint,
            "models": {n: s.to_dict() for n, s in self.models.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        """Deserializes a run state.

        Args:
            data (Dict[str, Any]): The serialized representation.

        Returns:
            RunState: The reconstructed run state.

        Raises:
            ValueError: If the manifest was written by a newer WarpRec version.
        """
        stored_version = data.get("schema_version", 0)
        if stored_version > SCHEMA_VERSION:
            raise ValueError(
                f"Run state schema version {stored_version} is newer than the "
                "version supported by this WarpRec installation "
                f"({SCHEMA_VERSION}). Upgrade WarpRec to resume this run."
            )
        state = cls(
            run_name=data["run_name"],
            pipeline=data["pipeline"],
            warprec_version=data["warprec_version"],
            writer_timestamp=data["writer_timestamp"],
            config_fingerprint=data["config_fingerprint"],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            schema_version=stored_version,
        )
        state.models = {
            n: ModelState.from_dict(s) for n, s in data.get("models", {}).items()
        }
        return state


def _json_default(value: Any) -> Any:
    """Converts values the JSON encoder cannot handle into native Python ones.

    Search algorithms hand back NumPy scalars rather than Python builtins, so a
    best-parameter block coming from a Bayesian strategy such as 'optuna',
    'hopt' or 'bohb' cannot be serialized as it is.

    Args:
        value (Any): The value the encoder could not serialize.

    Returns:
        Any: A JSON-serializable equivalent of the value.

    Raises:
        TypeError: If the value has no native equivalent.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _digest(payload: Any) -> str:
    """Computes a stable SHA-256 digest of a JSON-serializable payload.

    Args:
        payload (Any): The payload to digest.

    Returns:
        str: The hexadecimal digest.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _strip_excluded(model_params: Dict[str, Any]) -> Dict[str, Any]:
    """Removes fields that must not invalidate a resume from a parameter block.

    Args:
        model_params (Dict[str, Any]): The model parameter block.

    Returns:
        Dict[str, Any]: A copy without the excluded optimization fields.
    """
    cleaned = dict(model_params)
    optimization = cleaned.get("optimization")
    if isinstance(optimization, dict):
        cleaned["optimization"] = {
            k: v
            for k, v in optimization.items()
            if k not in _EXCLUDED_OPTIMIZATION_FIELDS
        }
    return cleaned


def model_fingerprint(model_name: str, model_params: Dict[str, Any]) -> str:
    """Fingerprints a single model's configuration.

    Resource requests and device selection are excluded so that a paused run can
    be resumed on a cluster of a different size.

    Args:
        model_name (str): The name of the model.
        model_params (Dict[str, Any]): The model's parameter block.

    Returns:
        str: The fingerprint.
    """
    return _digest({"model": model_name, "params": _strip_excluded(model_params)})


def run_fingerprint(config: "TrainConfiguration") -> str:
    """Fingerprints the run-level configuration.

    Covers everything that changes the data or the meaning of the results:
    reader, filtering, splitter, evaluation and the set of model names.

    Args:
        config (TrainConfiguration): The configuration of the experiment.

    Returns:
        str: The fingerprint.
    """
    return _digest(
        {
            "reader": config.reader.model_dump(),
            "filtering": config.filtering,
            "splitter": config.splitter.model_dump() if config.splitter else None,
            "evaluation": config.evaluation.model_dump(),
            "models": sorted(config.models.keys()),
        }
    )


def resolve_run_name(config: "TrainConfiguration") -> str:
    """Resolves the run name, deriving one when the configuration does not set it.

    Args:
        config (TrainConfiguration): The configuration of the experiment.

    Returns:
        str: The resolved run name.
    """
    if config.run.name:
        return config.run.name
    return f"{config.writer.dataset_name}_{run_fingerprint(config)[:8]}"


class RunStateStore:
    """Reads and writes the run state manifest through a WarpRec writer.

    Going through the writer means local and Azure Blob storage are both
    supported without any extra code, in the same way every other WarpRec
    artifact is written.

    Args:
        writer (Writer): The writer of the experiment.
        run_name (str): The identifier of the run.
    """

    def __init__(self, writer: "Writer", run_name: str):
        self._writer = writer
        self._run_name = run_name
        self._root = writer._path_join(writer.experiment_path, "run_state")  # pylint: disable=protected-access

    @property
    def state_path(self) -> str:
        """Returns the path of the run state manifest.

        Returns:
            str: The manifest path.
        """
        return self._writer._path_join(self._root, f"{self._run_name}.json")  # pylint: disable=protected-access

    def _eval_path(self, model_name: str) -> str:
        """Returns the path of a model's persisted evaluation results.

        Args:
            model_name (str): The name of the model.

        Returns:
            str: The path of the serialized results.
        """
        return self._writer._path_join(  # pylint: disable=protected-access
            self._root, self._run_name, "eval", f"{model_name}.pt"
        )

    def load(self) -> Optional[RunState]:
        """Loads the run state manifest.

        A manifest written by a newer WarpRec version makes 'RunState.from_dict'
        raise a ValueError, which is deliberately left to propagate: silently
        ignoring fields written by a newer version risks resuming into an
        inconsistent state.

        Returns:
            Optional[RunState]: The stored state, or None when it is absent or
                could not be parsed.
        """
        content = self._writer._read_text(self.state_path)  # pylint: disable=protected-access
        if not content:
            return None
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.attention(
                f"Run state at {self.state_path} could not be parsed ({e}). "
                "It will be treated as if no previous run existed."
            )
            return None
        return RunState.from_dict(data)

    def save(self, state: RunState) -> None:
        """Writes the run state manifest, replacing it entirely.

        Args:
            state (RunState): The state to persist.
        """
        state.updated_at = datetime.now().isoformat(timespec="seconds")
        self._writer._write_text(  # pylint: disable=protected-access
            self.state_path,
            json.dumps(state.to_dict(), indent=4, default=_json_default),
        )

    def save_eval_results(self, model_name: str, results: Dict[Any, Any]) -> None:
        """Persists a model's evaluation results for use by a resumed run.

        The per-user metric tensors are needed by the paired statistical
        significance tests, which compare every model in the experiment. Without
        them a resumed run could only compare the models it evaluated itself.

        Args:
            model_name (str): The name of the model.
            results (Dict[Any, Any]): The evaluation results to persist.
        """
        try:
            buffer = BytesIO()
            torch.save(results, buffer)
            buffer.seek(0)
            self._writer._write_bytes(  # pylint: disable=protected-access
                self._eval_path(model_name), buffer.read()
            )
        except (OSError, RuntimeError, ValueError) as e:
            logger.attention(
                f"Could not persist evaluation results for {model_name}: {e}. "
                "Statistical significance tests on a resumed run will exclude it."
            )

    def load_eval_results(self, model_name: str) -> Optional[Dict[Any, Any]]:
        """Loads a model's persisted evaluation results.

        Args:
            model_name (str): The name of the model.

        Returns:
            Optional[Dict[Any, Any]]: The stored results, or None when they are
                absent or cannot be read.
        """
        try:
            content = self._writer._read_bytes(self._eval_path(model_name))  # pylint: disable=protected-access
        except (OSError, RuntimeError) as e:
            logger.attention(
                f"Could not read evaluation results for {model_name}: {e}."
            )
            return None
        if not content:
            return None
        try:
            return torch.load(BytesIO(content), weights_only=False, map_location="cpu")
        except (RuntimeError, ValueError, EOFError) as e:
            logger.attention(
                f"Could not deserialize evaluation results for {model_name}: {e}."
            )
            return None
