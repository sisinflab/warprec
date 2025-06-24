from typing import List, Dict, Optional, Set

import torch
import re
from torch import Tensor
from pandas import DataFrame
from scipy.sparse import csr_matrix
from tabulate import tabulate
from math import ceil
from warprec.data.dataset import Dataset
from warprec.evaluation.metrics.base_metric import BaseMetric
from warprec.recommenders.base_recommender import Recommender
from warprec.utils.enums import MetricBlock, RecommenderModelType
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
        side_information (Optional[DataFrame]): The side information of the dataset.
        beta (float): The beta value used in some metrics.
        pop_ratio (float): The percentile considered popular.
        user_cluster (Tensor): The user cluster lookup tensor.
        item_cluster (Tensor): The item cluster lookup tensor.
    """

    def __init__(
        self,
        metric_list: List[str],
        k_values: List[int],
        train_set: csr_matrix,
        side_information: Optional[DataFrame],
        beta: float = 1.0,
        pop_ratio: float = 0.8,
        user_cluster: Tensor = None,
        item_cluster: Tensor = None,
    ):
        self.k_values = k_values
        self.metrics: Dict[int, List[BaseMetric]] = {}
        self.required_blocks: Dict[int, Set[MetricBlock]] = {}

        common_params = {
            "train_set": train_set,
            "side_information": side_information,
            "beta": beta,
            "pop_ratio": pop_ratio,
            "user_cluster": user_cluster,
            "item_cluster": item_cluster,
        }

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
                    **common_params,
                    **metric_params,  # Add specific params if needed
                )
                self.metrics[k].append(metric_instance)
                self.required_blocks[k].update(metric_instance._REQUIRED_COMPONENTS)

    def evaluate(
        self,
        model: Recommender,
        dataset: Dataset,
        device: str = "cpu",
        test_set: bool = True,
        verbose: bool = False,
    ):
        """The main method to evaluate a list of metrics on the prediction of a model.

        Args:
            model (Recommender): The trained model.
            dataset (Dataset): The dataset from which retrieve train/val/test data.
            device (str): The device on which the metrics will be calculated.
            test_set (bool): Wether or not to compute metrics on test set.
            verbose (bool): Wether of not the method should write with logger.
        """
        if verbose:
            partition = "test set" if test_set else "validation set"
            logger.separator()
            logger.msg(f"Starting evaluation for model {model.name} on {partition}.")

        # Reset all metrics in evaluator
        self.reset_metrics()
        model.eval()

        # Keep a copy of the full training set, if needed
        train_set = dataset.train_set

        # Iter over batches
        _start = 0
        for train_batch, test_batch, val_batch in dataset:
            _end = _start + train_batch.shape[0]  # Track strat - end of batch iteration
            current_users_idx_list = list(range(_start, _end))  # List of user idxs

            # If we are evaluating a sequential model, compute user history
            user_seq, seq_len = None, None
            if model.model_type == RecommenderModelType.SEQUENTIAL:
                user_seq, seq_len = train_set.get_user_history_sequences(
                    current_users_idx_list,
                    dataset.info()["max_seq_len"],  # Sequence length truncated
                )

            eval_set = test_batch if test_set else val_batch
            ground = torch.tensor(
                (eval_set).toarray(), device=device
            )  # Ground tensor [batch_size x items]

            predictions = model.predict(
                train_batch,
                start=_start,
                end=_end,
                train_set=train_set,
                user_seq=user_seq,
                seq_len=seq_len,
            ).to(device)  # Get ratings tensor [batch_size x items]

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
                        "ground": ground,
                        "binary_relevance": binary_relevance,
                        "discounted_relevance": discounted_relevance,
                        "valid_users": valid_users,
                        "start": _start,
                        **precomputed_blocks[k],
                    }
                    metric.update(
                        predictions,
                        **update_kwargs,
                    )
            _start = _end

        if verbose:
            logger.positive(f"Evaluation completed for model {model.name}.")

    def reset_metrics(self):
        """Reset all metrics accumulated values."""
        for metric_list in self.metrics.values():
            for metric in metric_list:
                metric.reset()

    def compute_results(self) -> Dict[int, Dict[str, float]]:
        """The method to retrieve computed results in dictionary format.

        Returns:
            Dict[int, Dict[str, float]]: The dictionary containing the results.
        """
        results: Dict[int, Dict[str, float]] = {}
        for k, metric_instances in self.metrics.items():
            results[k] = {}
            for metric in metric_instances:
                metric_result = metric.compute()
                results[k].update(metric_result)
        return results

    def print_console(
        self,
        res_dict: Dict[int, Dict[str, float]],
        header: str,
        max_metrics_per_row: int = 4,  # TODO: Add to config
    ):
        """Utility function to print results using tabulate.

        Args:
            res_dict (Dict[int, Dict[str, float]]): The dictionary containing all the results.
            header (str): The header of the evaluation grid,
                usually set with the name of evaluation.
            max_metrics_per_row (int): The number of metrics
                to print in each row.
        """
        # Collect all unique metric keys across all cutoffs
        first_cutoff_key = next(iter(res_dict))
        ordered_metric_keys = list(res_dict[first_cutoff_key].keys())

        # Split metric keys into chunks of size max_metrics_per_row
        n_chunks = ceil(len(ordered_metric_keys) / max_metrics_per_row)
        chunks = [
            ordered_metric_keys[i * max_metrics_per_row : (i + 1) * max_metrics_per_row]
            for i in range(n_chunks)
        ]

        # For each chunk, print a table with subset of metric columns
        for chunk_idx, chunk_keys in enumerate(chunks):
            _tab = []
            for k, metrics in res_dict.items():
                _metric_tab = [f"Top@{k}"]
                for key in chunk_keys:
                    _metric_tab.append(str(metrics.get(key, float("nan"))))
                _tab.append(_metric_tab)

            table = tabulate(
                _tab,
                headers=["Cutoff"] + chunk_keys,
                tablefmt="grid",
            )
            _rlen = len(table.split("\n", maxsplit=1)[0])
            title = header
            if n_chunks > 1:
                start_idx = chunk_idx * max_metrics_per_row + 1
                end_idx = min(
                    (chunk_idx + 1) * max_metrics_per_row, len(ordered_metric_keys)
                )
                title += f" (metrics {start_idx} - {end_idx})"
            logger.msg(title.center(_rlen, "-"))
            for row in table.split("\n"):
                logger.msg(row)
