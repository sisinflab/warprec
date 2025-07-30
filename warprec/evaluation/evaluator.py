import re
import time
from math import ceil
from typing import List, Dict, Optional, Set, Any

import torch
from torch import Tensor
from scipy.sparse import csr_matrix
from tabulate import tabulate
from warprec.data.dataset import Dataset
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
        self.metrics: Dict[str, Dict[int, List[BaseMetric]]] = {}
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

    def _init_metrics(self, set_type: str, metric_list: List[str], device: str):
        """Utility method to initialize metrics.

        Args:
            set_type (str): The type of set which requires the metrics to be set.
                Can be either 'test' or 'validation'.
            metric_list (List[str]): The list of metric names used to initialize
                metric classes from registry.
            device (str): The device where to compute the metric.

        Raises:
            ValueError: If the set_type is not 'test' or 'validation'.
        """
        if set_type not in ["test", "validation"]:
            raise ValueError("Unexpected set type during metrics initialization")

        self.metrics[set_type] = {}
        for k in self.k_values:
            self.metrics[set_type][k] = []
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
                ).to(device)
                self.metrics[set_type][k].append(metric_instance)
                self.required_blocks[k].update(metric_instance._REQUIRED_COMPONENTS)

    def evaluate(
        self,
        model: Recommender,
        dataset: Dataset,
        device: str = "cpu",
        evaluate_on_test: bool = False,
        evaluate_on_validation: bool = False,
        verbose: bool = False,
    ):
        """The main method to evaluate a list of metrics on the prediction of a model.

        Args:
            model (Recommender): The trained model.
            dataset (Dataset): The dataset from which retrieve train/val/test data.
            device (str): The device on which the metrics will be calculated.
            evaluate_on_test (bool): Wether or not to evaluate
                the model on test set.
            evaluate_on_validation (bool): Wether or not to evaluate
                the model on validation set.
            verbose (bool): Wether of not the method should write with logger.

        Raises:
            ValueError: If evaluation is required on a set not provided.
        """
        eval_start_time: float
        if verbose:
            logger.msg(f"Starting evaluation process for model {model.name}.")
            eval_start_time = time.time()

        # Initialize evaluation process
        self.reset_metrics()
        self.metrics_to(device)
        model.eval()

        # Extract main data structures from dataset
        train_set = dataset.train_set
        train_session = dataset.train_session

        # Initialize evaluation sets and inner data structures
        set_type_to_evaluate = []
        if evaluate_on_validation:
            if dataset.val_set is not None:
                self._init_metrics("validation", self.metric_list, device)
                set_type_to_evaluate.append("validation")
            else:
                raise ValueError(
                    "Evaluator is trying to evaluate on validation set, "
                    "but None have been provided."
                )
        if evaluate_on_test:
            if dataset.test_set is not None:
                self._init_metrics("test", self.metric_list, device)
                set_type_to_evaluate.append("test")
            else:
                raise ValueError(
                    "Evaluator is trying to evaluate on test set, "
                    "but None have been provided."
                )

        # Iter over batches
        _start = 0
        for train_batch, test_batch, val_batch in dataset:
            _end = _start + train_batch.shape[0]  # Track strat - end of batch iteration
            current_users_idx_list = list(range(_start, _end))  # List of user idxs
            user_indices = torch.tensor(current_users_idx_list, device=device)

            # If we are evaluating a sequential model, compute user history
            user_seq, seq_len = None, None
            if isinstance(model, SequentialRecommenderUtils):
                user_seq, seq_len = train_session.get_user_history_sequences(
                    current_users_idx_list,
                    model.max_seq_len,  # Sequence length truncated
                )

            predictions = model.predict(
                train_batch,
                start=_start,
                end=_end,
                user_indices=user_indices,
                train_set=train_set,
                user_seq=user_seq,
                user_id=torch.tensor(current_users_idx_list, dtype=torch.long),
                seq_len=seq_len,
            ).to(device)  # Get ratings tensor [batch_size x items]

            for set_type in set_type_to_evaluate:
                ground: Tensor
                if set_type == "test":
                    ground = torch.tensor(
                        (test_batch).toarray(), device=device
                    )  # Ground tensor [batch_size x items]
                elif set_type == "validation":
                    ground = torch.tensor(
                        (val_batch).toarray(), device=device
                    )  # Ground tensor [batch_size x items]
                else:
                    raise ValueError(
                        f"Set type error in evaluator. Set type received: {set_type}"
                    )

                # Pre-compute metric blocks
                precomputed_blocks: Dict[int, Dict[str, Tensor]] = {}

                # First we check for relevance
                binary_relevance = (
                    BaseMetric.binary_relevance(ground)
                    if MetricBlock.BINARY_RELEVANCE
                    in self.required_blocks[self.k_values[0]]
                    else None
                )
                discounted_relevance = (
                    BaseMetric.discounted_relevance(ground)
                    if MetricBlock.DISCOUNTED_RELEVANCE
                    in self.required_blocks[self.k_values[0]]
                    else None
                )

                # We also check if the number of valid users is required
                valid_users = (
                    BaseMetric.valid_users(ground)
                    if MetricBlock.VALID_USERS in self.required_blocks[self.k_values[0]]
                    else None
                )

                # Then we check all the needed blocks that are shared
                # between metrics so we can pre-computed
                for k in self.k_values:
                    precomputed_blocks[k] = {}
                    required_blocks_for_k = self.required_blocks[k]

                    # Check first if we need .top_k() method
                    if (
                        MetricBlock.TOP_K_VALUES
                        or MetricBlock.TOP_K_INDICES
                        or MetricBlock.TOP_K_BINARY_RELEVANCE
                        or MetricBlock.TOP_K_DISCOUNTED_RELEVANCE
                    ):
                        top_k_values, top_k_indices = BaseMetric.top_k_values_indices(
                            predictions, k
                        )
                        precomputed_blocks[k][f"top_{k}_values"] = top_k_values
                        precomputed_blocks[k][f"top_{k}_indices"] = top_k_indices

                        # Then we also check for .gather() method for better optimization
                        if MetricBlock.TOP_K_BINARY_RELEVANCE in required_blocks_for_k:
                            precomputed_blocks[k][f"top_{k}_binary_relevance"] = (
                                BaseMetric.top_k_relevance_from_indices(
                                    binary_relevance, top_k_indices
                                )
                            )
                        if (
                            MetricBlock.TOP_K_DISCOUNTED_RELEVANCE
                            in required_blocks_for_k
                        ):
                            precomputed_blocks[k][f"top_{k}_discounted_relevance"] = (
                                BaseMetric.top_k_relevance_from_indices(
                                    discounted_relevance, top_k_indices
                                )
                            )

                # Update all metrics on current batches
                for k, metric_instances in self.metrics[set_type].items():
                    for metric in metric_instances:
                        update_kwargs = {
                            "ground": ground,
                            "binary_relevance": binary_relevance,
                            "discounted_relevance": discounted_relevance,
                            "valid_users": valid_users,
                            "user_indices": user_indices,
                            "start": _start,
                            **precomputed_blocks[k],
                        }
                        metric.update(
                            predictions,
                            **update_kwargs,
                        )
            _start = _end

        if verbose:
            eval_total_time = time.time() - eval_start_time
            frmt_eval_total_time = time.strftime(
                "%H:%M:%S", time.gmtime(eval_total_time)
            )
            logger.positive(
                f"Evaluation completed for model {model.name}. Evaluation process took: {frmt_eval_total_time}"
            )

    def reset_metrics(self):
        """Reset all metrics accumulated values."""
        for k_metrics in self.metrics.values():
            for metric_list in k_metrics.values():
                for metric in metric_list:
                    metric.reset()

    def metrics_to(self, device: str):
        """Move all metrics to the same device.

        Args:
            device (str): The device where to move the metrics.
        """
        for k_metrics in self.metrics.values():
            for metric_list in k_metrics.values():
                for metric in metric_list:
                    metric.to(device)

    def compute_results(self) -> Dict[str, Dict[int, Dict[str, float | Tensor]]]:
        """The method to retrieve computed results in dictionary format.

        Returns:
            Dict[str, Dict[int, Dict[str, float | Tensor]]]: The dictionary containing the results.
        """
        results: Dict[str, Dict[int, Dict[str, float | Tensor]]] = {}
        for set_type, metric_dict in self.metrics.items():
            results[set_type] = {}
            for k, metric_instances in metric_dict.items():
                results[set_type][k] = {}
                for metric in metric_instances:
                    metric_result = metric.compute()
                    results[set_type][k].update(metric_result)
        return results

    def print_console(
        self,
        results: Dict[str, Dict[int, Dict[str, float | Tensor]]],
        max_metrics_per_row: int = 4,
    ):
        """Utility function to print results using tabulate.

        Args:
            results (Dict[str, Dict[int, Dict[str, float | Tensor]]]): The dictionary containing all the results.
            max_metrics_per_row (int): The number of metrics
                to print in each row.
        """
        for header, res_dict in results.items():
            # Collect all unique metric keys across all cutoffs
            first_cutoff_key = next(iter(res_dict))
            ordered_metric_keys = list(res_dict[first_cutoff_key].keys())

            # Split metric keys into chunks of size max_metrics_per_row
            n_chunks = ceil(len(ordered_metric_keys) / max_metrics_per_row)
            chunks = [
                ordered_metric_keys[
                    i * max_metrics_per_row : (i + 1) * max_metrics_per_row
                ]
                for i in range(n_chunks)
            ]

            # For each chunk, print a table with subset of metric columns
            for chunk_idx, chunk_keys in enumerate(chunks):
                _tab = []
                for k, metrics in res_dict.items():
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
