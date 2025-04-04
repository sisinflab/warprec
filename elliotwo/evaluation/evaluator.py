from typing import List, Dict, Union

import torch
from scipy.sparse import csr_matrix
from tabulate import tabulate
from elliotwo.data.dataset import AbstractDataset
from elliotwo.evaluation.base_metric import BaseMetric
from elliotwo.recommenders.abstract_recommender import AbstractRecommender
from elliotwo.utils.logger import logger
from elliotwo.utils.registry import metric_registry


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
    """

    def __init__(
        self,
        metric_list: List[str],
        k_values: List[int],
        train_set: csr_matrix,
        beta: float = 1.0,
        pop_ratio: float = 0.8,
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
                )
                for metric_name in metric_list
            ]
            for k in k_values
        }

    def evaluate(
        self,
        model: AbstractRecommender,
        dataset: AbstractDataset,
        device: str = "cpu",
        test_set: bool = True,
        verbose: bool = False,
    ):
        """The main method to evaluate a list of metrics on the prediction of a model.

        Args:
            model (AbstractRecommender): The trained model.
            dataset (AbstractDataset): The dataset from which retrieve train/val/test data.
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

        # Iter over batches
        _start = 0
        for train_batch, val_batch, test_batch in dataset:
            _end = _start + train_batch.shape[0]  # Track strat - end of batch iteration
            target = torch.tensor(
                (test_batch if test_set else val_batch).toarray(), device=device
            )  # Target tensor [batch_size x items]
            predictions = model.forward(train_batch, start=_start, end=_end).to(
                device
            )  # Get ratings tensor [batch_size x items]

            # Update all metrics on current batches
            for _, metric_instances in self.metrics.items():
                for metric in metric_instances:
                    metric.update(predictions, target)

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
        results = {}
        for k, metric_instances in self.metrics.items():
            results[k] = {
                metric.name: metric.compute().item() for metric in metric_instances
            }
        return results

    def print_console(
        self,
        res_dict: Dict[int, Dict[str, float]],
        metrics_list: List[str],
        header: str,
    ):
        """Utility function to print results using tabulate.

        Args:
            res_dict (Dict[int, Dict[str, float]]): The dictionary containing all the results.
            metrics_list (List[str]): The names of the metrics.
            header (str): The header of the evaluation grid,
                usually set with the name of evaluation.
        """
        _tab = []
        for k, metrics in res_dict.items():
            _metric_tab: List[Union[str, float]] = ["Top@" + str(k)]
            for _, score in metrics.items():
                _metric_tab.append(score)
            _tab.append(_metric_tab)
        table = tabulate(_tab, headers=metrics_list, tablefmt="grid")
        _rlen = len(table.split("\n", maxsplit=1)[0])
        logger.msg(header.center(_rlen, "-"))
        for row in table.split("\n"):
            logger.msg(row)
