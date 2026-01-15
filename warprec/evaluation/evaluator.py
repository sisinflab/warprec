import re
import time
from math import ceil
from typing import List, Dict, Optional, Set, Any, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix
from tabulate import tabulate

from warprec.data import Dataset
from warprec.evaluation.metrics.base_metric import BaseMetric
from warprec.recommenders.base_recommender import (
    Recommender,
    SequentialRecommenderUtils,
)
from warprec.utils.enums import MetricBlock
from warprec.utils.logger import logger
from warprec.utils.registry import metric_registry


class Evaluator:
    """Evaluator class will evaluate a trained model on a given
    set of metrics, taking into account the cutoff.

    Handles Full and Sampled evaluation on:
        - Collaborative Filtering models (+ content-base, + hybrids)
        - Context Aware models
        - Sequential models

    Args:
        metric_list (List[str]): The list of metric names that will
            be evaluated.
        k_values (List[int]): The cutoffs.
        train_set (csr_matrix): The train set sparse matrix.
        compute_per_user (bool): Wether or not to compute the metric
            per user or globally.
        beta (float): The beta value used in some metrics.
        pop_ratio (float): The percentile considered popular.
        feature_lookup (Optional[Tensor]): The feature lookup tensor.
        user_cluster (Optional[Tensor]): The user cluster lookup tensor.
        item_cluster (Optional[Tensor]): The item cluster lookup tensor.
        seed (int): The random seed for reproducibility.
    """

    def __init__(
        self,
        metric_list: List[str],
        k_values: List[int],
        train_set: csr_matrix,
        compute_per_user: bool = False,
        beta: float = 1.0,
        pop_ratio: float = 0.8,
        feature_lookup: Optional[Tensor] = None,
        user_cluster: Optional[Tensor] = None,
        item_cluster: Optional[Tensor] = None,
        seed: int = 42,
    ):
        self.k_values = k_values
        self.metric_list = metric_list
        self.num_items = train_set.shape[1]
        self.metrics: Dict[int, List[BaseMetric]] = {}
        self.required_blocks: Dict[int, Set[MetricBlock]] = {}

        # Set the seed for random permutation in sampled evaluation
        self.g = torch.Generator().manual_seed(seed)

        # Common parameters shared across metrics
        self.common_params: Dict[str, Any] = {
            "compute_per_user": compute_per_user,
            "num_users": train_set.shape[0],
            "num_items": train_set.shape[1],
            "item_interactions": torch.tensor(train_set.getnnz(axis=0)).float(),
            "item_indices": torch.tensor(train_set.indices, dtype=torch.long),
            "feature_lookup": feature_lookup,
            "beta": beta,
            "pop_ratio": pop_ratio,
            "user_cluster": user_cluster,
            "item_cluster": item_cluster,
        }
        self._init_metrics(metric_list)

    def _init_metrics(self, metric_list: List[str]):
        """Initializes metric instances from the registry."""
        for k in self.k_values:
            self.metrics[k] = []
            self.required_blocks[k] = set()
            for metric_string in metric_list:
                metric_name = metric_string
                metric_params = {}

                # Parsing complex metric names (e.g., F1[Precision, Recall])
                match_f1 = re.match(r"F1\[\s*(.*?)\s*,\s*(.*?)\s*\]", metric_string)
                match_efd_epc = re.match(r"(EFD|EPC)\[\s*(.*?)\s*\]", metric_string)

                if match_f1:
                    metric_name = "F1"
                    metric_params["metric_name_1"] = match_f1.group(1)
                    metric_params["metric_name_2"] = match_f1.group(2)

                if match_efd_epc:
                    metric_name = match_efd_epc.group(1)
                    metric_params["relevance"] = match_efd_epc.group(2)

                metric_instance = metric_registry.get(
                    metric_name,
                    k=k,
                    **self.common_params,
                    **metric_params,
                )
                self.metrics[k].append(metric_instance)
                self.required_blocks[k].update(metric_instance.components)

    def evaluate(
        self,
        model: Recommender,
        dataloader: DataLoader,
        strategy: str,
        dataset: Dataset,
        device: str = "cpu",
        verbose: bool = False,
    ):
        """Main evaluation loop.

        Args:
            model (Recommender): The model to evaluate.
            dataloader (DataLoader): The evaluation dataloader that will yield
                the eval set information.
            strategy (str): The strategy to use during evaluation.
            dataset (Dataset): The dataset object used for the evaluation.
            device (str): The device of the evaluation. Defaults to "cpu".
            verbose (bool): Wether or not to log progress during evaluation.

        Raises:
            ValueError: If the strategy isn't either "full" or "sampled".
        """
        # pylint: disable=too-many-statements
        if strategy not in ["full", "sampled"]:
            raise ValueError(f"Strategy '{strategy}' not supported.")

        if verbose:
            logger.msg(f"Starting evaluation process for model {model.name}.")
            eval_start_time = time.time()

        self.reset_metrics()
        self.metrics_to(device)
        model.eval()

        # Retrieve train interactions for masking (needed in full strategy)
        train_sparse = dataset.train_set.get_sparse()
        padding_idx = train_sparse.shape[1]

        for batch in dataloader:
            # Parse the batch
            batch_data = self._parse_batch(batch, strategy, device)

            eval_batch = None  # This will be the Binary Ground Truth for metrics

            user_indices = batch_data["user_indices"]
            context = batch_data.get("context")  # Optional (CARS only)
            candidates = batch_data.get("candidates")  # Optional (Sampled only)

            # Prepare the model input
            train_batch = train_sparse[user_indices.tolist(), :]
            predict_kwargs = {
                "user_indices": user_indices,
                "train_sparse": train_sparse,  # Some models might need it internally
                "train_batch": train_batch,
            }

            # A. Sequential Models Support
            if isinstance(model, SequentialRecommenderUtils):
                user_seq, seq_len = self._retrieve_sequences_for_user(
                    dataset, user_indices.tolist(), model.max_seq_len
                )
                predict_kwargs["user_seq"] = user_seq.to(device)
                predict_kwargs["seq_len"] = seq_len.to(device)

            # B. Context Support
            if context is not None:
                predict_kwargs["contexts"] = context

            # C. Item Indices (used in sampled evaluation)
            if strategy == "sampled":
                positives = batch_data["positives"]
                negatives = batch_data["negatives"]
                candidates = torch.cat([positives, negatives], dim=1)

                # Initialize the GT tensor
                eval_batch = torch.zeros_like(candidates)
                num_pos_cols = positives.shape[1]

                # Construct the GT based on the positives
                eval_batch[:, :num_pos_cols] = 1.0
                mask_padding = positives == padding_idx
                eval_batch[:, :num_pos_cols][mask_padding] = 0.0

                # Random permutation to avoid bias towards position
                perm = torch.randperm(candidates.shape[1], generator=self.g)
                candidates = candidates[:, perm]
                eval_batch = eval_batch[:, perm]

                predict_kwargs["item_indices"] = candidates

            # Model prediction
            predictions = model.predict(**predict_kwargs).to(device)

            if strategy == "full":
                if "target_item" in batch_data:
                    # Contextual full evaluation
                    target_item = batch_data["target_item"]
                    eval_batch = torch.zeros(
                        (len(user_indices), self.num_items), device=device
                    )
                    eval_batch.scatter_(1, target_item.unsqueeze(1), 1.0)
                else:
                    # Classic full evaluation
                    eval_batch = batch_data["ground_truth"]

                # Mask seen items
                predictions[train_batch.nonzero()] = -torch.inf

            elif strategy == "sampled":
                # Mask seen items
                predictions[candidates == padding_idx] = -torch.inf

            # Metric computation
            self._compute_metrics_step(
                predictions=predictions,
                eval_batch=eval_batch,
                user_indices=user_indices,
                candidates=candidates if strategy == "sampled" else None,
            )

        if verbose:
            self._log_results(eval_start_time, model.name)

    def _parse_batch(self, batch: Tuple, strategy: str, device: str) -> Dict[str, Any]:
        """Parses the batch tuple based on strategy and dimensions.

        Args:
            batch (Tuple): The batch to parse.
            strategy (str): The strategy used for evaluation.
            device (str): The device of the evaluation.

        Returns:
            Dict[str, Any]: Standardized dictionary of parsed parameters.

        Raises:
            ValueError: If the batch has unexpected length.
        """
        data = {}
        batch = [x.to(device) for x in batch]  # type: ignore[assignment]

        # Index 0 is ALWAYS user_indices
        data["user_indices"] = batch[0]

        if strategy == "full":
            if len(batch) == 2:
                # Standard: (users, ground_truth)
                data["ground_truth"] = batch[1]
            elif len(batch) == 3:
                # Contextual: (users, target_item, context)
                data["target_item"] = batch[1]
                data["context"] = batch[2]
            else:
                raise ValueError(
                    f"Unexpected batch size {len(batch)} for Full strategy"
                )

        elif strategy == "sampled":
            if len(batch) == 3:
                # Standard Sampled: (users, positives, negatives)
                data["positives"] = batch[1]
                data["negatives"] = batch[2]

            elif len(batch) == 4:
                # Contextual Sampled: (users, positives, negatives, context)
                data["positives"] = batch[1]
                data["negatives"] = batch[2]
                data["context"] = batch[3]
            else:
                raise ValueError(
                    f"Unexpected batch size {len(batch)} for Sampled strategy"
                )

        return data

    def _compute_metrics_step(self, predictions, eval_batch, user_indices, candidates):
        """Helper to isolate metric update logic."""

        # Pre-compute metric blocks (Optimization)
        precomputed_blocks: Dict[int, Dict[str, Tensor]] = {
            k: {} for k in self.k_values
        }
        all_required_blocks = set()
        for k in self.k_values:
            all_required_blocks.update(self.required_blocks.get(k, set()))

        # Relevance Computation
        binary_relevance = (
            BaseMetric.binary_relevance(eval_batch)
            if MetricBlock.BINARY_RELEVANCE in all_required_blocks
            else None
        )
        discounted_relevance = (
            BaseMetric.discounted_relevance(eval_batch)
            if MetricBlock.DISCOUNTED_RELEVANCE in all_required_blocks
            else None
        )

        # Top-K Computation (Once for max K)
        if self.k_values and any(
            b in all_required_blocks
            for b in [
                MetricBlock.TOP_K_VALUES,
                MetricBlock.TOP_K_INDICES,
                MetricBlock.TOP_K_BINARY_RELEVANCE,
                MetricBlock.TOP_K_DISCOUNTED_RELEVANCE,
            ]
        ):
            max_k = max(self.k_values)
            top_k_values_full, top_k_indices_full = BaseMetric.top_k_values_indices(
                predictions, max_k
            )

            for k in self.k_values:
                required = self.required_blocks.get(k, set())

                # Slicing
                top_k_indices = top_k_indices_full[:, :k]
                precomputed_blocks[k][f"top_{k}_values"] = top_k_values_full[:, :k]
                precomputed_blocks[k][f"top_{k}_indices"] = top_k_indices

                # Gathering Relevance
                if MetricBlock.TOP_K_BINARY_RELEVANCE in required:
                    precomputed_blocks[k][f"top_{k}_binary_relevance"] = (
                        BaseMetric.top_k_relevance_from_indices(
                            binary_relevance, top_k_indices
                        )
                    )

                if MetricBlock.TOP_K_DISCOUNTED_RELEVANCE in required:
                    precomputed_blocks[k][f"top_{k}_discounted_relevance"] = (
                        BaseMetric.top_k_relevance_from_indices(
                            discounted_relevance, top_k_indices
                        )
                    )

        # Update Metrics
        for k, metric_instances in self.metrics.items():
            for metric in metric_instances:
                update_kwargs = {
                    "ground": eval_batch,
                    "binary_relevance": binary_relevance,
                    "discounted_relevance": discounted_relevance,
                    "user_indices": user_indices,
                    "item_indices": candidates,  # None for Full, Tensor for Sampled
                    **precomputed_blocks[k],
                }
                metric.update(predictions, **update_kwargs)

    def _log_results(self, start_time, model_name):
        eval_total_time = time.time() - start_time
        frmt_time = time.strftime("%H:%M:%S", time.gmtime(eval_total_time))
        logger.positive(
            f"Evaluation completed for model {model_name}. Time: {frmt_time}"
        )

    def reset_metrics(self):
        """Reset all metrics accumulated values."""
        for metrics in self.metrics.values():
            for metric in metrics:
                metric.reset()

    def metrics_to(self, device: str):
        """Move all metrics to the same device.

        Args:
            device (str): The device where to move the metrics.
        """
        for metrics in self.metrics.values():
            for metric in metrics:
                metric.to(device)

    def compute_results(self) -> Dict[int, Dict[str, float | Tensor]]:
        """The method to retrieve computed results in dictionary format.

        Returns:
            Dict[int, Dict[str, float | Tensor]]: The dictionary containing the results.
        """
        results: Dict[int, Dict[str, float | Tensor]] = {}
        for k, metric_instances in self.metrics.items():
            results[k] = {}
            for metric in metric_instances:
                metric_result = metric.compute()
                results[k].update(metric_result)
        return results

    def print_console(
        self,
        results: Dict[int, Dict[str, float | Tensor]],
        header: str,
        max_metrics_per_row: int = 4,
    ):
        """Utility function to print results using tabulate.

        Args:
            results (Dict[int, Dict[str, float | Tensor]]): The dictionary containing
                all the results.
            header (str): The header of the table to be printed.
            max_metrics_per_row (int): The number of metrics
                to print in each row.
        """
        # Collect all unique metric keys across all cutoffs
        first_cutoff_key = next(iter(results))
        ordered_metric_keys = list(results[first_cutoff_key].keys())

        # Split metric keys into chunks of size max_metrics_per_row
        n_chunks = ceil(len(ordered_metric_keys) / max_metrics_per_row)
        chunks = [
            ordered_metric_keys[i * max_metrics_per_row : (i + 1) * max_metrics_per_row]
            for i in range(n_chunks)
        ]

        # For each chunk, print a table with subset of metric columns
        for chunk_idx, chunk_keys in enumerate(chunks):
            _tab = []
            for k, metrics in results.items():
                _metric_tab = [f"Top@{k}"]
                for key in chunk_keys:
                    metric = metrics.get(key, float("nan"))
                    if isinstance(
                        metric, Tensor
                    ):  # In case of user_wise computation, we compute the mean
                        metric = metric.mean().item()
                    _metric_tab.append(str(metric))
                _tab.append(_metric_tab)

            table = tabulate(
                _tab,
                headers=["Cutoff"] + chunk_keys,
                tablefmt="grid",
            )
            _rlen = len(table.split("\n", maxsplit=1)[0])
            title = header.capitalize()
            if n_chunks > 1:
                start_idx = chunk_idx * max_metrics_per_row + 1
                end_idx = min(
                    (chunk_idx + 1) * max_metrics_per_row, len(ordered_metric_keys)
                )
                title += f" (metrics {start_idx} - {end_idx})"
            logger.msg(title.center(_rlen, "-"))
            for row in table.split("\n"):
                logger.msg(row)

    def _retrieve_sequences_for_user(
        self, dataset: Dataset, user_indices: List[int], max_seq_len: int
    ) -> Tuple[Tensor, Tensor]:
        """Utility method to retrieve user sequences from dataset.

        Args:
            dataset (Dataset): The dataset containing the user sessions.
            user_indices (List[int]): The list of user indices.
            max_seq_len (int): The maximum sequence length.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing two elements:
                - Tensor: The user sequences.
                - Tensor: The lengths of the sequences.
        """
        # If we are evaluating a sequential model, compute user history
        user_seq, seq_len = dataset.train_session.get_user_history_sequences(
            user_indices,
            max_seq_len,  # Sequence length truncated
        )
        return (user_seq, seq_len)
