import math
import os
import time
from itertools import product
from typing import Any, Dict, List, Optional, Sequence

import lightning as L
import numpy as np
import psutil
import torch

from warprec.common import initialize_datasets
from warprec.data import Dataset
from warprec.data.reader import ReaderFactory
from warprec.data.writer import WriterFactory
from warprec.evaluation.evaluator import Evaluator
from warprec.recommenders.base_recommender import (
    IterativeRecommender,
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import (
    EstimateConfiguration,
    load_callback,
    load_estimate_configuration,
)
from warprec.utils.enums import SearchAlgorithms, SearchSpace
from warprec.utils.helpers import (
    build_evaluation_dataloader_kwargs,
    load_custom_modules,
    model_param_from_dict,
    resolve_num_workers,
    retrieve_evaluation_dataloader,
)
from warprec.utils.logger import logger
from warprec.utils.registry import model_registry


def _process_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def _normalize_device(device: Optional[str]) -> str:
    return "cpu" if device is None else str(device)


def _uses_cuda(device: str) -> bool:
    return device.startswith("cuda")


def _cuda_index(device: str) -> int:
    torch_device = torch.device("cuda:0" if device == "cuda" else device)
    return 0 if torch_device.index is None else torch_device.index


def _ensure_supported_device(device: str) -> None:
    if _uses_cuda(device) and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA probing requested on device '{device}', but CUDA is not available."
        )


def _sync_cuda(device: str) -> None:
    if _uses_cuda(device) and torch.cuda.is_available():
        torch.cuda.synchronize(_cuda_index(device))


def _reset_cuda_peak(device: str) -> None:
    if _uses_cuda(device) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(_cuda_index(device))


def _cuda_memory_allocated_mb(device: str) -> float:
    if not _uses_cuda(device) or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(_cuda_index(device)) / 1024**2


def _cuda_max_memory_allocated_mb(device: str) -> float:
    if not _uses_cuda(device) or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(_cuda_index(device)) / 1024**2


def _summarize_samples(
    samples: Sequence[float], max_override: Optional[float] = None
) -> Dict[str, float]:
    if not samples:
        base = [0.0]
    else:
        base = list(samples)

    arr = np.asarray(base, dtype=float)
    max_value = arr.max()
    if max_override is not None:
        max_value = max(max_value, float(max_override))

    return {
        "Min": float(arr.min()),
        "Avg": float(arr.mean()),
        "Max": float(max_value),
        "Std": float(arr.std(ddof=0)),
    }


def _mean_or_nan(values: Sequence[float]) -> float:
    filtered = [float(value) for value in values if not math.isnan(float(value))]
    if not filtered:
        return float("nan")
    return float(np.mean(filtered))


def _human_duration(seconds: float) -> str:
    if math.isnan(seconds):
        return "N/A"
    if seconds < 60:
        return f"{seconds:.3f} s"
    if seconds < 3600:
        return f"{seconds / 60:.3f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.3f} h"
    return f"{seconds / 86400:.3f} d"


def _human_memory(value_mb: float) -> str:
    if math.isnan(value_mb):
        return "N/A"

    value = float(value_mb) * 1024**2
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    while value >= 1024 and unit_idx < len(units) - 1:
        value /= 1024
        unit_idx += 1
    return f"{value:.3f} {units[unit_idx]}"


class EstimateStageTracker:
    """Tracks process-scoped RAM and optional CUDA memory during an estimate stage."""

    def __init__(self, baseline_rss_mb: float, device: str):
        self._baseline_rss_mb = baseline_rss_mb
        self._device = device
        self._ram_samples: List[float] = []
        self._vram_samples: List[float] = []

    def start(self) -> None:
        """Reset peak CUDA accounting and capture the initial sample."""
        _reset_cuda_peak(self._device)
        self.sample()

    def sample(self) -> None:
        """Capture the current RAM sample and, when available, the CUDA allocation."""
        self._ram_samples.append(max(_process_rss_mb() - self._baseline_rss_mb, 0.0))
        if _uses_cuda(self._device) and torch.cuda.is_available():
            self._vram_samples.append(_cuda_memory_allocated_mb(self._device))

    def summarize(self) -> Dict[str, Dict[str, float] | None]:
        """Return summary statistics for RAM and optional VRAM samples."""
        ram_stats = _summarize_samples(self._ram_samples)
        if not _uses_cuda(self._device) or not torch.cuda.is_available():
            return {"RAM": ram_stats, "VRAM": None}

        peak_vram_mb = _cuda_max_memory_allocated_mb(self._device)
        vram_stats = _summarize_samples(self._vram_samples, max_override=peak_vram_mb)
        return {"RAM": ram_stats, "VRAM": vram_stats}


class EstimateTrainingCallback(L.Callback):
    """Collects training batch timings and resource samples from Lightning."""

    def __init__(
        self,
        warmup_batches: int,
        measured_batches: int,
        tracker: EstimateStageTracker,
        device: str,
    ):
        super().__init__()
        self._warmup_batches = warmup_batches
        self._measured_batches = measured_batches
        self._tracker = tracker
        self._device = device
        self._current_start: Optional[float] = None
        self._seen_batches = 0
        self.batch_times: List[float] = []

    def on_fit_start(self, trainer, pl_module) -> None:
        self._tracker.start()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._tracker.sample()
        _sync_cuda(self._device)
        self._current_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        _sync_cuda(self._device)
        elapsed = 0.0
        if self._current_start is not None:
            elapsed = time.perf_counter() - self._current_start

        if self._seen_batches >= self._warmup_batches:
            if len(self.batch_times) < self._measured_batches:
                self.batch_times.append(elapsed)

        self._seen_batches += 1
        self._tracker.sample()


def _model_device(config: EstimateConfiguration, setup: dict) -> str:
    general_device = _normalize_device(config.general.device)
    model_params = setup.get("optimization", {})
    return (
        _normalize_device(model_params.get("device"))
        if model_params.get("device") is not None
        else general_device
    )


def _training_dataset(
    main_dataset: Dataset,
    val_dataset: Optional[Dataset],
    fold_dataset: List[Dataset],
) -> Dataset:
    if val_dataset is not None:
        return val_dataset
    if fold_dataset:
        return main_dataset
    return main_dataset


def _lightning_accelerator(device: str) -> str:
    return "gpu" if _uses_cuda(device) else "cpu"


def _lightning_devices(device: str) -> int:
    return 1 if _uses_cuda(device) else 1


def _setup_options(field: str, value: Any) -> List[Any]:
    if not isinstance(value, list):
        return [value]
    if len(value) == 0:
        raise ValueError(f"Model field '{field}' cannot define an empty search space.")

    space = value[0]
    if isinstance(space, SearchSpace):
        space = space.value

    if space not in {SearchSpace.GRID.value, SearchSpace.CHOICE.value}:
        raise ValueError(
            f"Estimate only supports discrete grid-like spaces. "
            f"Field '{field}' uses unsupported search space '{space}'."
        )

    options = value[1:]
    if not options:
        raise ValueError(f"Model field '{field}' has no values to estimate.")
    return options


def _expand_model_setups(model_name: str, model_params: dict) -> List[dict]:
    optimization = model_params.get("optimization", {})
    strategy = optimization.get("strategy", SearchAlgorithms.GRID)
    if isinstance(strategy, SearchAlgorithms):
        strategy = strategy.value
    if strategy != SearchAlgorithms.GRID.value:
        raise ValueError(
            f"Estimate only supports '{SearchAlgorithms.GRID.value}' optimization, "
            f"but model '{model_name}' uses '{strategy}'."
        )

    param_fields = [
        field
        for field in model_params.keys()
        if field not in {"meta", "optimization", "early_stopping"}
    ]
    field_values = [
        _setup_options(field, model_params[field]) for field in param_fields
    ]

    setups = []
    for values in product(*field_values):
        setup = {}
        for optional_field in ("meta", "optimization", "early_stopping"):
            if (
                optional_field in model_params
                and model_params[optional_field] is not None
            ):
                setup[optional_field] = model_params[optional_field]
        setup.update(dict(zip(param_fields, values)))
        setups.append(setup)
    return setups


def _create_evaluator(dataset: Dataset, config: EstimateConfiguration) -> Evaluator:
    return Evaluator(
        list(config.evaluation.metrics),
        list(config.evaluation.top_k),
        train_set=dataset.train_set.get_sparse(),
        additional_data=dataset.get_stash(),
        complex_metrics=config.evaluation.complex_metrics,
        feature_lookup=dataset.get_features_lookup(),
        user_cluster=dataset.get_user_cluster(),
        item_cluster=dataset.get_item_cluster(),
    )


def _estimate_eval_loop(
    evaluator: Evaluator,
    model: Recommender,
    dataloader,
    strategy: str,
    dataset: Dataset,
    device: str,
    warmup_batches: int,
    measured_batches: int,
    tracker: EstimateStageTracker,
) -> Dict[str, Any]:
    # pylint: disable=too-many-statements
    evaluator.reset_metrics()
    evaluator.metrics_to(device)
    model.eval()

    tracker.start()
    batch_times: List[float] = []
    train_sparse = dataset.train_set.get_sparse()
    padding_idx = train_sparse.shape[1]
    max_batches = warmup_batches + measured_batches

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        tracker.sample()
        _sync_cuda(device)
        start = time.perf_counter()

        batch_data = evaluator._parse_batch(  # pylint: disable=protected-access
            batch, strategy, device
        )
        eval_batch = None
        user_indices = batch_data["user_indices"]
        context = batch_data.get("context")
        candidates = batch_data.get("candidates")

        train_batch = train_sparse[user_indices.tolist(), :]
        predict_kwargs = {"user_indices": user_indices}

        if isinstance(model, SequentialRecommenderUtils):
            user_seq, seq_len = evaluator._retrieve_sequences_for_user(  # pylint: disable=protected-access
                dataset, user_indices.tolist(), model.max_seq_len
            )
            predict_kwargs["user_seq"] = user_seq.to(device)
            predict_kwargs["seq_len"] = seq_len.to(device)

        if context is not None:
            predict_kwargs["contexts"] = context

        if strategy == "sampled":
            positives = batch_data["positives"]
            negatives = batch_data["negatives"]
            candidates = torch.cat([positives, negatives], dim=1)

            eval_batch = torch.zeros_like(candidates)
            num_pos_cols = positives.shape[1]
            eval_batch[:, :num_pos_cols] = 1.0
            mask_padding = positives == padding_idx
            eval_batch[:, :num_pos_cols][mask_padding] = 0.0

            perm = torch.randperm(candidates.shape[1], generator=evaluator.g)
            candidates = candidates[:, perm]
            eval_batch = eval_batch[:, perm]
            predict_kwargs["item_indices"] = candidates

        with torch.inference_mode():
            predictions = model.predict(**predict_kwargs).to(device)

            if strategy == "full":
                if "target_item" in batch_data:
                    target_item = batch_data["target_item"]
                    eval_batch = torch.zeros(
                        (len(user_indices), evaluator.num_items), device=device
                    )
                    eval_batch.scatter_(1, target_item.unsqueeze(1), 1.0)
                else:
                    eval_batch = batch_data["ground_truth"]

                predictions[train_batch.nonzero()] = -torch.inf
            else:
                predictions[candidates == padding_idx] = -torch.inf

            evaluator._compute_metrics_step(  # pylint: disable=protected-access
                predictions=predictions,
                eval_batch=eval_batch,
                user_indices=user_indices,
                candidates=candidates if strategy == "sampled" else None,
            )

        _sync_cuda(device)
        elapsed = time.perf_counter() - start
        if batch_idx >= warmup_batches:
            batch_times.append(elapsed)
        tracker.sample()

    results = evaluator.compute_results()
    return {
        "batch_times": batch_times,
        "measured_batches": len(batch_times),
        "results": results,
        "resources": tracker.summarize(),
    }


def _build_model(
    model_name: str,
    setup: dict,
    dataset: Dataset,
    seed: int,
    block_size: int,
    chunk_size: int,
) -> Recommender:
    return model_registry.get(
        name=model_name,
        params=setup,
        interactions=dataset.train_set,
        sessions=dataset.train_session,
        seed=seed,
        info=dataset.info(),
        **dataset.get_stash(),
        block_size=block_size,
        chunk_size=chunk_size,
    )


def _analytical_report(train_ram_mb: float, notes: List[str]) -> Dict[str, Any]:
    return {
        "Measured Train Batches": 0,
        "Measured Eval Batches": 0,
        "Train Batch Time Avg": float("nan"),
        "Train Batch Time Std": float("nan"),
        "Estimated Train Epoch Time": float("nan"),
        "Estimated Total Train Time": float("nan"),
        "Eval Batch Time Avg": float("nan"),
        "Eval Batch Time Std": float("nan"),
        "Estimated Evaluation Time": float("nan"),
        "Train RAM Min": train_ram_mb,
        "Train RAM Avg": train_ram_mb,
        "Train RAM Max": train_ram_mb,
        "Train RAM Std": 0.0,
        "Eval RAM Min": float("nan"),
        "Eval RAM Avg": float("nan"),
        "Eval RAM Max": float("nan"),
        "Eval RAM Std": float("nan"),
        "Train VRAM Min": float("nan"),
        "Train VRAM Avg": float("nan"),
        "Train VRAM Max": float("nan"),
        "Train VRAM Std": float("nan"),
        "Eval VRAM Min": float("nan"),
        "Eval VRAM Avg": float("nan"),
        "Eval VRAM Max": float("nan"),
        "Eval VRAM Std": float("nan"),
        "Notes": "; ".join(dict.fromkeys(notes)),
    }


def _require_numeric(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"Field '{field_name}' must be numeric, got {type(value).__name__}."
        )
    return float(value)


def _run_estimate_setup(
    config: EstimateConfiguration,
    model_name: str,
    setup: dict,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    estimate_settings: dict,
) -> Dict[str, Any]:
    # pylint: disable=too-many-statements
    load_custom_modules(config.general.custom_modules)

    params = model_param_from_dict(model_name, setup)
    seed = params.optimization.properties.seed
    block_size = params.optimization.block_size
    chunk_size = params.optimization.chunk_size
    device = _model_device(config, setup)
    _ensure_supported_device(device)

    model_class = model_registry.get_class(model_name)
    is_iterative_model = issubclass(model_class, IterativeRecommender)

    notes: List[str] = []
    if not _uses_cuda(device):
        notes.append("CPU only")

    if not is_iterative_model:
        estimate = getattr(model_class, "estimate_space")(
            params=setup,
            info=train_dataset.info(),
            interactions=train_dataset.train_set,
            sessions=train_dataset.train_session,
            block_size=block_size,
            chunk_size=chunk_size,
            **train_dataset.get_stash(),
        )
        notes.append("Analytical train-space estimate")
        if estimate.get("notes"):
            notes.append(str(estimate["notes"]))
        return _analytical_report(
            train_ram_mb=float(estimate["train_ram_mb"]),
            notes=notes,
        )

    baseline_rss_mb = _process_rss_mb()
    train_tracker = EstimateStageTracker(baseline_rss_mb=baseline_rss_mb, device=device)
    callback: WarpRecCallback = load_callback(
        config.general.callback,
        *config.general.callback.args,
        **config.general.callback.kwargs,
    )

    model = _build_model(
        model_name=model_name,
        setup=setup,
        dataset=train_dataset,
        seed=seed,
        block_size=block_size,
        chunk_size=chunk_size,
    )

    train_time_avg = float("nan")
    train_time_std = float("nan")
    estimated_train_epoch_time = float("nan")
    estimated_total_train_time = float("nan")
    measured_train_batches = 0

    if isinstance(model, IterativeRecommender):
        model.set_optimization_parameters(
            optimizer_config=params.optimization.optimizer,
            lr_scheduler_config=params.optimization.lr_scheduler,
        )

        # Keep estimate dataloader startup cheap and deterministic.
        # The goal is to profile the model itself rather than worker spin-up cost.
        num_workers = 1
        persistent_workers = False
        pin_memory = _uses_cuda(device)

        train_dataloader = model.get_dataloader(
            interactions=train_dataset.train_set,
            sessions=train_dataset.train_session,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        total_train_batches = len(train_dataloader)
        train_limit = min(
            estimate_settings["warmup_batches"] + estimate_settings["train_batches"],
            total_train_batches,
        )

        training_callback = EstimateTrainingCallback(
            warmup_batches=estimate_settings["warmup_batches"],
            measured_batches=estimate_settings["train_batches"],
            tracker=train_tracker,
            device=device,
        )
        trainer = L.Trainer(
            max_epochs=1,
            devices=_lightning_devices(device),
            accelerator=_lightning_accelerator(device),
            limit_train_batches=train_limit,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            limit_val_batches=0,
            callbacks=[training_callback],
        )
        trainer.fit(model, train_dataloaders=train_dataloader)

        measured_train_batches = len(training_callback.batch_times)
        if training_callback.batch_times:
            train_time_avg = float(np.mean(training_callback.batch_times))
            train_time_std = float(np.std(training_callback.batch_times))
            estimated_train_epoch_time = train_time_avg * total_train_batches
        estimated_total_train_time = (
            estimated_train_epoch_time * model.epochs
            if not math.isnan(estimated_train_epoch_time)
            else float("nan")
        )

    train_resources = train_tracker.summarize()

    callback.on_training_complete(model=model)

    eval_num_workers = resolve_num_workers(
        params.optimization.num_workers,
        os.cpu_count(),
    )
    evaluation_dataloader_kwargs = build_evaluation_dataloader_kwargs(
        num_workers=eval_num_workers,
        device=device,
        reuse_loader=False,
    )
    eval_dataloader = retrieve_evaluation_dataloader(
        dataset=eval_dataset,
        model=model,
        strategy=config.evaluation.strategy,
        num_negatives=config.evaluation.num_negatives,
        **evaluation_dataloader_kwargs,
    )
    model.to(device)

    eval_tracker = EstimateStageTracker(baseline_rss_mb=baseline_rss_mb, device=device)
    evaluator = _create_evaluator(eval_dataset, config)
    eval_estimate = _estimate_eval_loop(
        evaluator=evaluator,
        model=model,
        dataloader=eval_dataloader,
        strategy=config.evaluation.strategy,
        dataset=eval_dataset,
        device=device,
        warmup_batches=estimate_settings["warmup_batches"],
        measured_batches=estimate_settings["eval_batches"],
        tracker=eval_tracker,
    )
    eval_times = eval_estimate["batch_times"]
    eval_time_avg = float(np.mean(eval_times)) if eval_times else float("nan")
    eval_time_std = float(np.std(eval_times)) if eval_times else float("nan")
    estimated_eval_time = (
        eval_time_avg * len(eval_dataloader) if eval_times else float("nan")
    )

    callback.on_evaluation_complete(
        model=model,
        params=setup,
        results=eval_estimate["results"],
    )

    report = {
        "Measured Train Batches": measured_train_batches,
        "Measured Eval Batches": eval_estimate["measured_batches"],
        "Train Batch Time Avg": train_time_avg,
        "Train Batch Time Std": train_time_std,
        "Estimated Train Epoch Time": estimated_train_epoch_time,
        "Estimated Total Train Time": estimated_total_train_time,
        "Eval Batch Time Avg": eval_time_avg,
        "Eval Batch Time Std": eval_time_std,
        "Estimated Evaluation Time": estimated_eval_time,
        "Train RAM Min": train_resources["RAM"]["Min"],
        "Train RAM Avg": train_resources["RAM"]["Avg"],
        "Train RAM Max": train_resources["RAM"]["Max"],
        "Train RAM Std": train_resources["RAM"]["Std"],
        "Eval RAM Min": eval_estimate["resources"]["RAM"]["Min"],
        "Eval RAM Avg": eval_estimate["resources"]["RAM"]["Avg"],
        "Eval RAM Max": eval_estimate["resources"]["RAM"]["Max"],
        "Eval RAM Std": eval_estimate["resources"]["RAM"]["Std"],
        "Train VRAM Min": float("nan"),
        "Train VRAM Avg": float("nan"),
        "Train VRAM Max": float("nan"),
        "Train VRAM Std": float("nan"),
        "Eval VRAM Min": float("nan"),
        "Eval VRAM Avg": float("nan"),
        "Eval VRAM Max": float("nan"),
        "Eval VRAM Std": float("nan"),
        "Notes": "; ".join(dict.fromkeys(notes)),
    }

    if train_resources["VRAM"] is not None:
        report.update(
            {
                "Train VRAM Min": train_resources["VRAM"]["Min"],
                "Train VRAM Avg": train_resources["VRAM"]["Avg"],
                "Train VRAM Max": train_resources["VRAM"]["Max"],
                "Train VRAM Std": train_resources["VRAM"]["Std"],
            }
        )

    if eval_estimate["resources"]["VRAM"] is not None:
        report.update(
            {
                "Eval VRAM Min": eval_estimate["resources"]["VRAM"]["Min"],
                "Eval VRAM Avg": eval_estimate["resources"]["VRAM"]["Avg"],
                "Eval VRAM Max": eval_estimate["resources"]["VRAM"]["Max"],
                "Eval VRAM Std": eval_estimate["resources"]["VRAM"]["Std"],
            }
        )

    if _uses_cuda(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return report


def _run_setup(
    config: EstimateConfiguration,
    model_name: str,
    setup: dict,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    estimate_settings: dict,
) -> Dict[str, Any]:
    return _run_estimate_setup(
        config=config,
        model_name=model_name,
        setup=setup,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        estimate_settings=estimate_settings,
    )


def _aggregate_reports(
    model_name: str, setup_reports: List[Dict[str, Any]]
) -> Dict[str, Any]:
    numeric_keys = [
        key
        for key, value in setup_reports[0].items()
        if isinstance(value, (int, float))
    ]
    aggregated = {"Model Name": model_name, "Setup Count": len(setup_reports)}

    for key in numeric_keys:
        aggregated[key] = _mean_or_nan([report[key] for report in setup_reports])

    aggregated["Measured Train Batches"] = int(
        round(
            _require_numeric(
                aggregated["Measured Train Batches"], "Measured Train Batches"
            )
        )
    )
    aggregated["Measured Eval Batches"] = int(
        round(
            _require_numeric(
                aggregated["Measured Eval Batches"], "Measured Eval Batches"
            )
        )
    )

    notes = [
        report["Notes"]
        for report in setup_reports
        if isinstance(report.get("Notes"), str) and report["Notes"]
    ]
    aggregated["Notes"] = "; ".join(dict.fromkeys(notes))
    return aggregated


def _log_estimate_summary(model_report: Dict[str, Any]) -> None:
    logger.msg(
        f"Estimate summary for {model_report['Model Name']}: "
        f"setups={model_report['Setup Count']}, "
        f"est_train={_human_duration(model_report['Estimated Total Train Time'])}, "
        f"est_eval={_human_duration(model_report['Estimated Evaluation Time'])}"
    )
    logger.msg(
        f"Train RAM max={_human_memory(model_report['Train RAM Max'])}, "
        f"Eval RAM max={_human_memory(model_report['Eval RAM Max'])}, "
        f"Train VRAM max={_human_memory(model_report['Train VRAM Max'])}, "
        f"Eval VRAM max={_human_memory(model_report['Eval VRAM Max'])}"
    )


def estimate_pipeline(path: str) -> None:
    """Main function to start the estimate pipeline.

    Args:
        path (str): Path to the configuration file.

    Raises:
        ValueError: If the configured split output format is not supported.
    """
    logger.attention(
        "WARNING: Estimate pipeline is experimental. Please submit bug reports via GitHub Issues."
    )
    logger.msg("Starting the Estimate Pipeline.")
    experiment_start_time = time.time()

    config = load_estimate_configuration(path)
    callback: WarpRecCallback = load_callback(
        config.general.callback,
        *config.general.callback.args,
        **config.general.callback.kwargs,
    )

    reader = ReaderFactory.get_reader(config=config)
    writer = WriterFactory.get_writer(config=config)

    main_dataset, val_dataset, fold_dataset = initialize_datasets(
        reader=reader,
        callback=callback,
        config=config,
    )

    if config.splitter and config.writer.save_split:
        file_format = config.writer.split.file_format
        match file_format:
            case "tabular":
                writer.write_tabular_split(
                    main_dataset,
                    val_dataset,
                    fold_dataset,
                    **config.writer.split.model_dump(),
                )
            case "parquet":
                writer.write_parquet_split(
                    main_dataset,
                    val_dataset,
                    fold_dataset,
                    **config.writer.split.model_dump(),
                )
            case _:
                raise ValueError(f"File format '{file_format}'not supported.")

    data_preparation_time = time.time() - experiment_start_time
    logger.positive(
        f"Data preparation completed in {data_preparation_time:.2f} seconds."
    )

    train_dataset = _training_dataset(main_dataset, val_dataset, fold_dataset)
    estimate_settings = config.estimate.model_dump()

    for model_name, model_params in config.models.items():
        logger.msg(f"Estimating model {model_name}.")
        setups = _expand_model_setups(model_name, model_params)
        setup_reports = []

        for idx, setup in enumerate(setups, start=1):
            logger.msg(
                f"Running estimate setup {idx}/{len(setups)} for model {model_name}."
            )
            try:
                setup_report = _run_setup(
                    config=config,
                    model_name=model_name,
                    setup=setup,
                    train_dataset=train_dataset,
                    eval_dataset=main_dataset,
                    estimate_settings=estimate_settings,
                )
                setup_reports.append(setup_report)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.negative(
                    f"Estimate setup {idx}/{len(setups)} for model "
                    f"{model_name} failed: {exc}"
                )

        if not setup_reports:
            logger.negative(
                f"No successful estimate setups completed for model {model_name}."
            )
            continue

        model_report = _aggregate_reports(model_name, setup_reports)
        _log_estimate_summary(model_report)
        writer.write_estimate_report(
            [model_report],
            **config.writer.results.model_dump(),
        )

    logger.positive(
        "Estimate pipeline executed successfully. WarpRec is shutting down."
    )
