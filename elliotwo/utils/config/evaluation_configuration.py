from typing import List, Optional

from pydantic import BaseModel, field_validator
from elliotwo.utils.registry import metric_registry


class EvaluationConfig(BaseModel):
    """Definition of Evaluation configuration.

    Attributes:
        top_k (List[int]): List of cutoffs to evaluate.
        metrics (List[str]): List of metrics to compute during evaluation.
        beta (Optional[float]): The beta value used in some metrics like F1 score.
        save_evaluation (Optional[bool]): Wether or not to save the evaluation.
    """

    top_k: List[int]
    metrics: List[str]
    beta: Optional[float] = 1.0
    save_evaluation: Optional[bool] = True

    @field_validator("metrics")
    @classmethod
    def metrics_validator(cls, v: List[str]):
        """Validate metrics."""
        for metric in v:
            if metric.upper() not in metric_registry.list_registered():
                raise ValueError(
                    f"Metric {metric} not in metric registry. This is the list"
                    f"of supported metrics: {metric_registry.list_registered()}"
                )
        return v
