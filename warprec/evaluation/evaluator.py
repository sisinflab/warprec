import re
import time
from math import ceil
from typing import List, Dict, Optional, Set, Any, Tuple

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from tabulate import tabulate
from warprec.data import (
    Dataset,
    EvaluationDataLoader,
    NegativeEvaluationDataLoader,
)
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

    If a validation set has been provided in the dataset, then this
    class will provide results on validation too.

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
    ):
        self.k_values = k_values
        self.metric_list = metric_list
        self.num_items = train_set.shape[1]
        self.metrics: Dict[int, List[BaseMetric]] = {}
        self.required_blocks: Dict[int, Set[MetricBlock]] = {}
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
        """Utility method to initialize metrics.

        Args:
            metric_list (List[str]): The list of metric names used to initialize
                metric classes from registry.
        """
        for k in self.k_values:
            self.metrics[k] = []
            self.required_blocks[k] = set()
            for metric_string in metric_list:
                metric_name = metric_string
                metric_params = {}

                # Check for F1-extended
                match_f1 = re.match(r"F1\[\s*(.*?)\s*,\s*(.*?)\s*\]", metric_string)
                match_efd_epc = re.match(r"(EFD|EPC)\[\s*(.*?)\s*\]", metric_string)

                if match_f1:
                    metric_name = "F1"  # Generic name for F1-Extended

                    # Retrieve sub metric names
                    # validation has been done inside Pydantic
                    metric_params["metric_name_1"] = match_f1.group(1)
                    metric_params["metric_name_2"] = match_f1.group(2)

                if match_efd_epc:
                    metric_name = match_efd_epc.group(1)

                    # Retrieve relevance
                    # validation has been done inside Pydantic
                    metric_params["relevance"] = match_efd_epc.group(2)

                # Generic metric initialization
                metric_instance = metric_registry.get(
                    metric_name,
                    k=k,
                    **self.common_params,
                    **metric_params,  # Add specific params if needed
                )
                self.metrics[k].append(metric_instance)
                self.required_blocks[k].update(metric_instance._REQUIRED_COMPONENTS)

    def evaluate(
        self,
        model: Recommender,
        dataloader: EvaluationDataLoader | NegativeEvaluationDataLoader,
        strategy: str,
        dataset: Dataset,
        device: str = "cpu",
        verbose: bool = False,
    ):
        """The main method to evaluate a list of metrics on the prediction of a model.

        Args:
            model (Recommender): The trained model.
            dataloader (EvaluationDataLoader | NegativeEvaluationDataLoader):
                The dataloader used for the evaluation.
            strategy (str): The evaluation strategy to use.
            dataset (Dataset): The dataset used for the evaluation.
            device (str): The device on which the metrics will be calculated.
            verbose (bool): Wether of not the method should write with logger.

        Raises:
            ValueError: If the strategy is not supported.
        """
        # Check if the strategy is correct
        if strategy not in ["full", "sampled"]:
            raise ValueError(
                "The strategy passed is not correct. "
                "Accepted strategies are 'full' and 'sampled'. "
                f"Strategy received: {strategy}"
            )

        eval_start_time: float
        if verbose:
            logger.msg(f"Starting evaluation process for model {model.name}.")
            eval_start_time = time.time()

        # Initialize evaluation process
        self.reset_metrics()
        self.metrics_to(device)
        model.eval()

        # Set the train interactions
        train_sparse = dataset.train_set.get_sparse()
        padding_idx = train_sparse.shape[1]

        # Main evaluation loop
        for batch in dataloader:
            candidates_local: Tensor = None
            train_batch: csr_matrix = None

            # Based on strategy, call different predict method
            match strategy:
                case "full":
                    eval_batch, user_indices = [x.to(device) for x in batch]

                    # Index the interactions of the current users
                    train_batch = train_sparse[user_indices.tolist(), :]

                    # In case of sequential model, we need to retrieve sequences
                    user_seq, seq_len = None, None
                    if isinstance(model, SequentialRecommenderUtils):
                        user_seq, seq_len = self._retrieve_sequences_for_user(
                            dataset,
                            user_indices.tolist(),
                            model.max_seq_len,
                        )

                    predictions = model.predict(
                        user_indices=user_indices,
                        user_seq=user_seq,
                        seq_len=seq_len,
                        train_batch=train_batch,
                        train_sparse=train_sparse,
                    ).to(device)  # Get ratings tensor [batch_size, num_items]

                    # Masking interaction already seen in train
                    predictions[train_batch.nonzero()] = -torch.inf
                case "sampled":
                    pos_batch, neg_batch, user_indices = [x.to(device) for x in batch]

                    # Index the interactions of the current users
                    train_batch = train_sparse[user_indices.tolist(), :]

                    # In case of sequential model, we need to retrieve sequences
                    user_seq, seq_len = None, None
                    if isinstance(model, SequentialRecommenderUtils):
                        user_seq, seq_len = self._retrieve_sequences_for_user(
                            dataset,
                            user_indices.tolist(),
                            model.max_seq_len,
                        )

                    # Cat all the sampled items in a single tensor
                    candidates_local = torch.cat([pos_batch, neg_batch], dim=1)

                    # This method will rate only sampled items
                    # Output tensor size will depend on longest sampled
                    # list in current batch
                    predictions = model.predict(
                        user_indices=user_indices,
                        item_indices=candidates_local,
                        user_seq=user_seq,
                        seq_len=seq_len,
                        train_batch=train_batch,
                        train_sparse=train_sparse,
                    ).to(device)  # Get ratings tensor [batch_size, pad_seq]

                    # Mask padded indices
                    predictions[candidates_local == padding_idx] = -torch.inf

                    # Create the local GT
                    num_positives_per_user = (pos_batch != padding_idx).sum(dim=1)
                    col_indices = torch.arange(candidates_local.shape[1], device=device)
                    eval_batch = (
                        col_indices < num_positives_per_user.unsqueeze(1)
                    ).float()  # [batch_size, pad_seq]

            # Pre-compute metric blocks
            precomputed_blocks: Dict[int, Dict[str, Tensor]] = {
                k: {} for k in self.k_values
            }
            all_required_blocks = set()
            for k in self.k_values:
                all_required_blocks.update(self.required_blocks.get(k, set()))

            # First we check for relevance
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

            # We also check if the number of valid users is required
            valid_users = (
                BaseMetric.valid_users(eval_batch)
                if MetricBlock.VALID_USERS in all_required_blocks
                else None
            )

            # Efficiently compute top_k once for the maximum k
            if self.k_values and (
                MetricBlock.TOP_K_VALUES in all_required_blocks
                or MetricBlock.TOP_K_INDICES in all_required_blocks
                or MetricBlock.TOP_K_BINARY_RELEVANCE in all_required_blocks
                or MetricBlock.TOP_K_DISCOUNTED_RELEVANCE in all_required_blocks
            ):
                max_k = max(self.k_values)
                top_k_values_full, top_k_indices_full = BaseMetric.top_k_values_indices(
                    predictions, max_k
                )

                # Then we check all the needed blocks that are shared
                # between metrics so we can pre-compute by slicing
                for k in self.k_values:
                    required_blocks_for_k = self.required_blocks.get(k, set())

                    top_k_values = top_k_values_full[:, :k]
                    top_k_indices = top_k_indices_full[:, :k]

                    precomputed_blocks[k][f"top_{k}_values"] = top_k_values
                    precomputed_blocks[k][f"top_{k}_indices"] = top_k_indices

                    # Then we also check for .gather() method for better optimization
                    if MetricBlock.TOP_K_BINARY_RELEVANCE in required_blocks_for_k:
                        precomputed_blocks[k][f"top_{k}_binary_relevance"] = (
                            BaseMetric.top_k_relevance_from_indices(
                                binary_relevance, top_k_indices
                            )
                        )
                    if MetricBlock.TOP_K_DISCOUNTED_RELEVANCE in required_blocks_for_k:
                        precomputed_blocks[k][f"top_{k}_discounted_relevance"] = (
                            BaseMetric.top_k_relevance_from_indices(
                                discounted_relevance, top_k_indices
                            )
                        )

            # Update all metrics on current batches
            for k, metric_instances in self.metrics.items():
                for metric in metric_instances:
                    update_kwargs = {
                        "ground": eval_batch,
                        "binary_relevance": binary_relevance,
                        "discounted_relevance": discounted_relevance,
                        "valid_users": valid_users,
                        "user_indices": user_indices,
                        "item_indices": candidates_local,
                        **precomputed_blocks[k],
                    }
                    metric.update(
                        predictions,
                        **update_kwargs,
                    )

            # Manual garbage collection for heavy data
            del (
                predictions,
                candidates_local,
                eval_batch,
                binary_relevance,
                discounted_relevance,
                valid_users,
            )
            if "top_k_values_full" in locals():
                del top_k_values_full, top_k_indices_full, top_k_values, top_k_indices
            if "precomputed_blocks" in locals():
                del precomputed_blocks
            if "batch" in locals():
                del batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if verbose:
            eval_total_time = time.time() - eval_start_time
            frmt_eval_total_time = time.strftime(
                "%H:%M:%S", time.gmtime(eval_total_time)
            )
            logger.positive(
                f"Evaluation completed for model {model.name}. "
                f"Evaluation process took: {frmt_eval_total_time}"
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
