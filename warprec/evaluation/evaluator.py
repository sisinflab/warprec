from typing import List, Dict

import torch
from scipy.sparse import csr_matrix
from tabulate import tabulate
from math import ceil
from warprec.data.dataset import Dataset
from warprec.evaluation.base_metric import BaseMetric
from warprec.recommenders.base_recommender import Recommender
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
        beta (float): The beta value used in some metrics.
        pop_ratio (float): The percentile considered popular.
        user_cluster (Dict[int, int]): The user cluster mapping.
        item_cluster (Dict[int, int]): The item cluster mapping.
    """

    def __init__(
        self,
        metric_list: List[str],
        k_values: List[int],
        train_set: csr_matrix,
        beta: float = 1.0,
        pop_ratio: float = 0.8,
        user_cluster: Dict[int, int] = None,
        item_cluster: Dict[int, int] = None,
    ):
        self.k_values = k_values
        self.metrics: Dict[int, List[BaseMetric]] = {
            k: [
                metric_registry.get(
                    metric_name,
                    k=k,
                    train_set=train_set,
                    beta=beta,
                    pop_ratio=pop_ratio,
                    user_cluster=user_cluster,
                    item_cluster=item_cluster,
                )
                for metric_name in metric_list
            ]
            for k in k_values
        }

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

        # Iter over batches
        _start = 0
        for train_batch, test_batch, val_batch in dataset:
            _end = _start + train_batch.shape[0]  # Track strat - end of batch iteration
            eval_set = test_batch if test_set else val_batch
            target = torch.tensor(
                (eval_set).toarray(), device=device
            )  # Target tensor [batch_size x items]

            predictions = model.predict(train_batch, start=_start, end=_end).to(
                device
            )  # Get ratings tensor [batch_size x items]

            # Update all metrics on current batches
            for _, metric_instances in self.metrics.items():
                for metric in metric_instances:
                    metric.update(predictions, target, start=_start)

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
                if isinstance(metric_result, dict):
                    # Merge dict entries into results
                    results[k].update(metric_result)
                else:
                    # Single scalar value
                    results[k][metric.name] = metric_result.item()
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
        all_metric_keys: set[str] = set()
        for metrics in res_dict.values():
            all_metric_keys.update(metrics.keys())
        sorted_metric_keys = sorted(all_metric_keys)

        # Split metric keys into chunks of size max_metrics_per_row
        n_chunks = ceil(len(sorted_metric_keys) / max_metrics_per_row)
        chunks = [
            sorted_metric_keys[i * max_metrics_per_row : (i + 1) * max_metrics_per_row]
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
                    (chunk_idx + 1) * max_metrics_per_row, len(all_metric_keys)
                )
                title += f" (metrics {start_idx} - {end_idx})"
            logger.msg(title.center(_rlen, "-"))
            for row in table.split("\n"):
                logger.msg(row)
