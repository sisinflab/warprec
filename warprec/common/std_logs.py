from math import ceil
from typing import Dict

from torch import Tensor
from tabulate import tabulate

from warprec.utils.logger import logger


def log_evaluation(
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
                    metric = metric.nanmean().item()
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
