import os
import time
from typing import Tuple, Dict, List

import ray
import lightning as L
import torch
from torch import Tensor

from warprec.data import Dataset
from warprec.utils.helpers import (
    retrieve_evaluation_dataloader,
)
from warprec.recommenders.base_recommender import (
    Recommender,
    IterativeRecommender,
    SequentialRecommenderUtils,
    ContextRecommenderUtils,
)
from warprec.evaluation import Evaluator
from warprec.utils.helpers import load_custom_modules
from warprec.utils.config import RecomModel
from warprec.utils.config.evaluation_configuration import ComplexMetricConfig
from warprec.utils.registry import model_registry


@ray.remote
def remote_evaluation_and_timing(
    model: Recommender,
    main_dataset: Dataset,
    metrics: List[str],
    top_k: List[int],
    complex_metrics: List[ComplexMetricConfig],
    strategy: str,
    num_negatives: int,
    device: str,
    requires_timing: bool,
    custom_modules: List[str],
) -> Tuple[Dict[int, Dict[str, float | Tensor]], float, float]:
    """This remote function will be executed on the Ray Cluster using requested resources.

    This will offload heavy computation to other nodes which ensuring fast execution times.

    Args:
        model (Recommender): The trained model to evaluate.
        main_dataset (Dataset): The dataset on which evaluate the model.
        metrics (List[str]): The name of the metrics to compute.
        top_k (List[int]): The cutoff values.
        complex_metrics (List[ComplexMetricConfig]): The configuration of the complex metrics.
        strategy (str): The evaluation strategy.
        num_negatives (int): The number of negative samples to use with 'sampled' strategy.
        device (str): The device to use for evaluation.
        requires_timing (bool): Wether or not to calculate timings.
        custom_modules (List[str]): The list of custom modules to load on the worker node.

    Returns:
        Tuple[Dict[int, Dict[str, float | Tensor]], float, float]: A tuple containing:
            - Dict[int, Dict[str, float | Tensor]]: The results of the evaluation.
            - float: The evaluation timing.
            - float: The inference timing.
    """
    # Load custom modules if provided
    load_custom_modules(custom_modules)

    # Move to worker node device
    model.to(device)

    # Force model to use the same seed after de-serialization on the worker node
    model.set_seed(model.seed)

    # Instantiate the Evaluator locally on the worker node
    evaluator = Evaluator(
        metric_list=metrics,
        k_values=top_k,
        train_set=main_dataset.train_set.get_sparse(),
        additional_data=main_dataset.get_stash(),
        complex_metrics=complex_metrics,
        feature_lookup=main_dataset.get_features_lookup(),
        user_cluster=main_dataset.get_user_cluster(),
        item_cluster=main_dataset.get_item_cluster(),
    )

    # Retrieve dataloader
    dataloader = retrieve_evaluation_dataloader(
        dataset=main_dataset,
        model=model,
        strategy=strategy,
        num_negatives=num_negatives,
    )

    # Evaluation
    eval_start_time = time.time()
    evaluator.evaluate(
        model=model,
        dataloader=dataloader,
        strategy=strategy,
        dataset=main_dataset,
        device=device,
        verbose=True,
    )
    results = evaluator.compute_results()
    eval_time = time.time() - eval_start_time

    # Move metrics back to CPU to avoid CUDA errors on the driver node
    for k, metric_dict in results.items():
        for metric_name, value in metric_dict.items():
            if isinstance(value, Tensor):
                results[k][metric_name] = value.cpu()

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

        inference_time_start = time.time()
        with torch.inference_mode():
            model.predict(
                user_indices=user_indices,
                item_indices=item_indices,
                user_seq=user_seq,
                seq_len=seq_len,
                contexts=contexts,
            )
        inference_time = time.time() - inference_time_start

    return results, eval_time, inference_time


@ray.remote
def remote_model_retraining(
    model_name: str,
    best_params: dict,
    main_dataset: Dataset,
    params: RecomModel,
    custom_modules: List[str],
    device: str,
    seed: int,
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

    Returns:
        Tuple[Recommender, dict, int]:
            - The trained model.
            - A report dictionary with parameter counts.
            - The number of iterations.
    """
    # Load custom modules if provided
    load_custom_modules(custom_modules)

    block_size = params.optimization.block_size
    chunk_size = params.optimization.chunk_size
    num_workers = params.optimization.num_workers
    iterations = best_params["iterations"]

    # Retrieve the model from the registry
    best_model = model_registry.get(
        name=model_name,
        params=best_params,
        interactions=main_dataset.train_set,
        sessions=main_dataset.train_session,
        seed=seed,
        info=main_dataset.info(),
        **main_dataset.get_stash(),
        block_size=block_size,
        chunk_size=chunk_size,
    )

    # Train the model using backpropagation if the model is iterative
    if isinstance(best_model, IterativeRecommender):
        best_model.set_optimization_parameters(
            optimizer_config=params.optimization.optimizer,
            lr_scheduler_config=params.optimization.lr_scheduler,
        )

        if num_workers is None:
            available_cpus = os.cpu_count()
            num_workers = max(available_cpus - 1, 1)

        persistent_workers = num_workers > 0
        pin_memory = device == "cuda"

        train_dataloader = best_model.get_dataloader(
            interactions=main_dataset.train_set,
            sessions=main_dataset.train_session,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        l_trainer = L.Trainer(
            max_epochs=best_model.epochs,
            devices="auto",
            accelerator=device,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        l_trainer.fit(
            best_model,
            train_dataloaders=train_dataloader,
        )

    # Final reporting
    report = {}
    report["Total Params (Best Model)"] = sum(
        p.numel() for p in best_model.parameters()
    )
    report["Trainable Params (Best Model)"] = sum(
        p.numel() for p in best_model.parameters() if p.requires_grad
    )

    return best_model, report, iterations
