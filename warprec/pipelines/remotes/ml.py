import time
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Tuple, Dict, List

import ray
import lightning as L
import torch
from torch import Tensor

from warprec.data.stash import MODEL_ARGUMENTS, check_stash
from warprec.data import Dataset
from warprec.data.writer import Writer
from warprec.pipelines.remotes.data import generate_recs, write_model_artifacts
from warprec.recommenders.reranking import Reranker, build_reranker
from warprec.recommenders.trainer import Trainer, TrainingOutcome
from warprec.evaluation import build_evaluator
from warprec.utils.helpers import (
    build_evaluation_dataloader_kwargs,
    resolve_available_cpus,
    resolve_num_workers,
    retrieve_evaluation_dataloader,
)
from warprec.recommenders.trainer.runtime import lightning_runtime
from warprec.recommenders.trainer.timing import (
    EPOCH_TRAIN_S,
    RETRAINING_TRAIN,
    EpochTimingCallback,
    synchronize,
)
from warprec.recommenders.base_recommender import (
    Recommender,
    IterativeRecommender,
    SequentialRecommenderUtils,
    ContextRecommenderUtils,
)
from warprec.utils.determinism import apply_determinism
from warprec.utils.helpers import load_custom_modules
from warprec.utils.config import RecomModel, TrainConfiguration
from warprec.utils.config.evaluation_configuration import EvaluationConfig
from warprec.utils.registry import model_registry


@ray.remote  # type: ignore[arg-type]
def remote_evaluation_and_timing(
    model: Recommender,
    main_dataset: Dataset,
    evaluation: EvaluationConfig,
    num_workers: Optional[int],
    device: str,
    requires_timing: bool,
    custom_modules: List[str],
    gpu_deterministic: bool = False,
) -> Tuple[Dict[int, Dict[str, float | Tensor]], float, float]:
    """This remote function will be executed on the Ray Cluster using requested resources.

    This will offload heavy computation to other nodes which ensuring fast execution times.

    Args:
        model (Recommender): The trained model to evaluate.
        main_dataset (Dataset): The dataset on which evaluate the model.
        evaluation (EvaluationConfig): The evaluation section of the configuration.
        num_workers (Optional[int]): The number of dataloader workers to use for evaluation.
        device (str): The device to use for evaluation.
        requires_timing (bool): Wether or not to calculate timings.
        custom_modules (List[str]): The list of custom modules to load on the worker node.
        gpu_deterministic (bool): Whether to use deterministic algorithms only.

    Returns:
        Tuple[Dict[int, Dict[str, float | Tensor]], float, float]: What
            'evaluate_and_time' returns.
    """
    # Load custom modules if provided
    load_custom_modules(custom_modules)
    apply_determinism(gpu_deterministic)
    return evaluate_and_time(
        model, main_dataset, evaluation, num_workers, device, requires_timing
    )


def evaluate_and_time(
    model: Recommender,
    main_dataset: Dataset,
    evaluation: EvaluationConfig,
    num_workers: Optional[int],
    device: str,
    requires_timing: bool,
    reranker: Optional[Reranker] = None,
) -> Tuple[Dict[int, Dict[str, float | Tensor]], float, float]:
    """Evaluate a model on the test set, and time one prediction call.

    Args:
        model (Recommender): The trained model to evaluate.
        main_dataset (Dataset): The dataset on which evaluate the model.
        evaluation (EvaluationConfig): The evaluation section of the configuration.
        num_workers (Optional[int]): The number of dataloader workers to use for evaluation.
        device (str): The device to use for evaluation.
        requires_timing (bool): Wether or not to calculate timings.
        reranker (Optional[Reranker]): The re-ranker applied to every list, the
            one the recommendations are written with.

    Returns:
        Tuple[Dict[int, Dict[str, float | Tensor]], float, float]: A tuple containing:
            - Dict[int, Dict[str, float | Tensor]]: The results of the evaluation.
            - float: The seconds the evaluation of the model on the test set
                took on this worker: scoring every batch of the loader and
                computing the metrics. Ray's scheduling of this task, moving
                the model here and building the loader are not part of it.
            - float: The seconds of one prediction call scoring up to 1,000
                users against up to 1,000 items each, 0 when no timing is
                required.
    """
    # Move to worker node device
    model.to(device)

    # Force model to use the same seed after de-serialization on the worker node
    model.set_seed(model.seed)

    # Instantiate the Evaluator locally on the worker node
    evaluator = build_evaluator(evaluation, main_dataset, reranker)

    # Retrieve dataloader
    evaluation_dataloader_kwargs = build_evaluation_dataloader_kwargs(
        num_workers=resolve_num_workers(num_workers, resolve_available_cpus()),
        device=device,
        reuse_loader=False,
    )
    dataloader = retrieve_evaluation_dataloader(
        dataset=main_dataset,
        model=model,
        strategy=evaluation.strategy,
        num_negatives=evaluation.num_negatives,
        seed=evaluation.seed,
        negative_sampling=evaluation.negative_sampling,
        neg_alpha=evaluation.neg_alpha,
        **evaluation_dataloader_kwargs,
    )

    # Evaluation
    synchronize(device)
    eval_start_time = time.perf_counter()
    evaluator.evaluate(
        model=model,
        dataloader=dataloader,
        strategy=evaluation.strategy,
        dataset=main_dataset,
        device=device,
        verbose=True,
    )
    results = evaluator.compute_results()
    synchronize(device)
    eval_time = time.perf_counter() - eval_start_time

    # Move metrics back to CPU to avoid CUDA errors on the driver node
    for metric_dict in results.values():
        for metric_name, value in metric_dict.items():
            if isinstance(value, Tensor):
                metric_dict[metric_name] = value.cpu()

    # Time Report (if requested)
    inference_time = 0.0
    if requires_timing:
        info = main_dataset.info()
        n_users = info.get("n_users", 1000)
        n_items = info.get("n_items", 1000)
        context_dims = info.get("context_dims", {})

        n_users_to_predict = min(1000, n_users)
        n_items_to_predict = min(1000, n_items)

        max_seq_len = (
            model.max_seq_len if isinstance(model, SequentialRecommenderUtils) else 10
        )

        contexts = None
        if isinstance(model, ContextRecommenderUtils):
            model_labels = model.context_labels
            if model_labels:
                ctx_list = []
                for label in model_labels:
                    dim = context_dims.get(label, 10)
                    c_data = torch.randint(1, dim, (n_users_to_predict,)).to(
                        device=device
                    )
                    ctx_list.append(c_data)
                contexts = torch.stack(ctx_list, dim=1)

        user_indices = torch.arange(n_users_to_predict).to(device=device)
        item_indices = torch.randint(
            1, n_items, (n_users_to_predict, n_items_to_predict)
        ).to(device=device)
        user_seq = torch.randint(1, n_items, (n_users_to_predict, max_seq_len)).to(
            device=device
        )
        seq_len = torch.randint(1, max_seq_len + 1, (n_users_to_predict,)).to(
            device=device
        )

        synchronize(device)
        inference_time_start = time.perf_counter()
        with torch.inference_mode():
            model.predict(
                user_indices=user_indices,
                item_indices=item_indices,
                user_seq=user_seq,
                seq_len=seq_len,
                contexts=contexts,
            )
        synchronize(device)
        inference_time = time.perf_counter() - inference_time_start

    return results, eval_time, inference_time


@ray.remote  # type: ignore[arg-type]
def remote_model_retraining(
    model_name: str,
    best_params: dict,
    main_dataset: Dataset,
    params: RecomModel,
    custom_modules: List[str],
    device: str,
    seed: int,
    gpu_deterministic: bool = False,
) -> Tuple[Recommender, dict, int]:
    """Remote task to handle the final model retraining on a worker node.

    This prevents the driver node from executing PyTorch Lightning training loops,
    which could lead to CPU/GPU starvation or Out-Of-Memory errors.

    Args:
        model_name (str): The name of the model.
        best_params (dict): The best hyperparameters found during HPO.
        main_dataset (Dataset): The dataset to train on.
        params (RecomModel): The model configuration parameters.
        custom_modules (List[str]): The list of custom modules to load on the worker node.
        device (str): The device to use for training.
        seed (int): The random seed.
        gpu_deterministic (bool): Whether to use deterministic algorithms only.

    Returns:
        Tuple[Recommender, dict, int]: What 'retrain_model' returns.
    """
    # Load custom modules if provided
    load_custom_modules(custom_modules)
    apply_determinism(gpu_deterministic)
    return retrain_model(model_name, best_params, main_dataset, params, device, seed)


def retrain_model(
    model_name: str,
    best_params: dict,
    main_dataset: Dataset,
    params: RecomModel,
    device: str,
    seed: int,
) -> Tuple[Recommender, dict, int]:
    """Retrain a cross-validated model on the main split, as many epochs as chosen.

    Args:
        model_name (str): The name of the model.
        best_params (dict): The best hyperparameters found during HPO.
        main_dataset (Dataset): The dataset to train on.
        params (RecomModel): The model configuration parameters.
        device (str): The device to use for training.
        seed (int): The random seed.

    Returns:
        Tuple[Recommender, dict, int]:
            - The trained model.
            - A report dictionary with parameter counts and the seconds the
              retraining spent only training.
            - The number of iterations.
    """
    block_size = params.optimization.block_size
    chunk_size = params.optimization.chunk_size
    num_workers = params.optimization.num_workers
    iterations = best_params["iterations"]

    # Retrieve the model from the registry. A closed-form model is fitted here
    build_start = time.perf_counter()
    best_model = model_registry.get(
        name=model_name,
        params=best_params,
        interactions=main_dataset.train_set,
        transactions=main_dataset.train_transactions,
        knowledge=main_dataset.knowledge,
        multimodal=main_dataset.multimodal,
        sessions=main_dataset.train_session,
        seed=seed,
        info=main_dataset.info(),
        **check_stash(main_dataset.get_stash(), MODEL_ARGUMENTS, "model"),
        block_size=block_size,
        chunk_size=chunk_size,
    )
    synchronize(device)
    train_only_s = time.perf_counter() - build_start

    # Train the model using backpropagation if the model is iterative
    if isinstance(best_model, IterativeRecommender):
        best_model.set_optimization_parameters(
            optimizer_config=params.optimization.optimizer,
            lr_scheduler_config=params.optimization.lr_scheduler,
        )

        if num_workers is None:
            available_cpus = resolve_available_cpus()
            num_workers = max(available_cpus - 1, 1)

        persistent_workers = num_workers > 0
        pin_memory = device == "cuda"

        train_dataloader = best_model.get_dataloader(
            interactions=main_dataset.train_set,
            sessions=main_dataset.train_session,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            seed=best_model.seed,
        )

        # The epochs cross-validation settled on, not the search's upper bound
        timing = EpochTimingCallback()
        l_trainer = L.Trainer(
            max_epochs=iterations,
            devices="auto",
            accelerator=device,
            **lightning_runtime(params.optimization, device),
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            callbacks=[timing],
        )
        l_trainer.fit(
            best_model,
            train_dataloaders=train_dataloader,
        )
        train_only_s = sum(epoch[EPOCH_TRAIN_S] for epoch in timing.epochs)

        # What is saved with the model, and written as its best parameters,
        # has to describe the training it had, not the search's upper bound
        best_model.epochs = iterations

    # Final reporting
    report: Dict[str, float] = {}
    report["Total Params (Best Model)"] = sum(
        p.numel() for p in best_model.parameters()
    )
    report["Trainable Params (Best Model)"] = sum(
        p.numel() for p in best_model.parameters() if p.requires_grad
    )
    report[RETRAINING_TRAIN] = train_only_s

    return best_model, report, iterations


@dataclass
class ModelCompletion:
    """What the task completing a model hands back to the driver.

    Attributes:
        outcome (TrainingOutcome): The outcome of the search, its report given
            the best model's parameter counts and, after cross-validation, the
            epochs it was retrained for. It carries no model.
        error (Optional[str]): Why the best model could not be built, None when
            it was.
        results (Dict[int, Dict[str, float | Tensor]]): The test results.
        best_params (Dict[str, Any]): The hyperparameters of the best model.
        build_seconds (float): The seconds spent building the best model: loading
            it, refitting it, or retraining it after cross-validation.
        evaluation_seconds (float): The seconds the test evaluation took.
        inference_seconds (float): The seconds of one prediction call.
        model (Optional[Recommender]): The best model, on the CPU, only when the
            driver asked for it.
    """

    outcome: TrainingOutcome
    error: Optional[str] = None
    results: Dict[int, Dict[str, float | Tensor]] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)
    build_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    inference_seconds: float = 0.0
    model: Optional[Recommender] = None


@ray.remote  # type: ignore[arg-type]
def remote_complete_model(
    model_name: str,
    params: RecomModel,
    outcome: TrainingOutcome,
    main_dataset: Dataset,
    config: TrainConfiguration,
    writer: Writer,
    cross_validated: bool,
    device: str,
    return_model: bool = False,
) -> ModelCompletion:
    """Build, evaluate and write the best model of a finished search.

    Everything that needs the model or the training data runs here, on the
    cluster, so that the driver holds neither: the best model is loaded from
    its trial checkpoint, refitted when it is closed-form or retrained after
    cross-validation, then evaluated on the test set, and its per-user results,
    its serialized checkpoint and its recommendation lists are written by this
    worker.

    Args:
        model_name (str): The name of the model.
        params (RecomModel): The parameters of the model.
        outcome (TrainingOutcome): The completed outcome of its search.
        main_dataset (Dataset): The main dataset, the train/test split.
        config (TrainConfiguration): The configuration object.
        writer (Writer): The writer of the run.
        cross_validated (bool): Whether the search ran over folds.
        device (str): The device to build and evaluate the model on.
        return_model (bool): Whether to hand the model back, for a callback that
            receives it.

    Returns:
        ModelCompletion: The outcome, the results and the timings, or the error
            that kept the best model from being built.
    """
    load_custom_modules(config.general.custom_modules)
    apply_determinism(config.general.gpu_deterministic)

    build_start = time.perf_counter()
    try:
        if cross_validated:
            model, retrain_report, iterations = retrain_model(
                model_name,
                outcome.best_params or {},
                main_dataset,
                params,
                device,
                params.optimization.properties.seed,
            )
            outcome = replace(
                outcome,
                report={**outcome.report, **retrain_report},
                best_iter=iterations,
            )
        else:
            model = Trainer.load_best_model(
                model_name,
                str(outcome.best_checkpoint_path),
                outcome.best_params or {},
                main_dataset,
            )
            outcome = replace(
                outcome,
                report={**outcome.report, **Trainer.parameter_counts(model)},
            )
    except Exception as e:
        return ModelCompletion(outcome=outcome, error=f"{type(e).__name__}: {e}")
    build_seconds = time.perf_counter() - build_start

    results, evaluation_seconds, inference_seconds = evaluate_and_time(
        model,
        main_dataset,
        config.evaluation,
        params.optimization.num_workers,
        device,
        config.general.time_report,
        # The lists evaluated are the lists written
        build_reranker(config.rerank, main_dataset),
    )
    if params.meta.save_recs:
        generate_recs(writer, model, main_dataset, config, device)

    # Written and handed back from the CPU, which every reader has
    model.cpu()
    write_model_artifacts(
        writer, config, model_name, params, model, results, main_dataset
    )

    return ModelCompletion(
        outcome=outcome,
        results=results,
        best_params=model.get_params(),
        build_seconds=build_seconds,
        evaluation_seconds=evaluation_seconds,
        inference_seconds=inference_seconds,
        model=model if return_model else None,
    )
