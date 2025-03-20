from typing import List, Optional

from pydantic import BaseModel, field_validator
from elliotwo.utils.registry import metric_registry


class EvaluationConfig(BaseModel):
    """Definition of Evaluation configuration.

    Attributes:
        top_k (List[int]): List of cutoffs to evaluate.
        metrics (List[str]): List of metrics to compute during evaluation.
        beta (Optional[float]): The beta value used in some metrics like F1 score.
        pop_ratio (Optional[float]): The percentage of item transactions that
            will be considered popular.
        save_evaluation (Optional[bool]): Wether or not to save the evaluation.
    """

    top_k: List[int]
    metrics: List[str]
    beta: Optional[float] = 1.0
    pop_ratio: Optional[float] = 0.8
    save_evaluation: Optional[bool] = True

    @field_validator("top_k")
    @classmethod
    def top_k_validator(cls, v: List[int]):
        """Validate top_k."""
        for k in v:
            if k <= 0:
                raise ValueError("Values for top_k should be positive integers.")
        return v

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

    @field_validator("beta")
    @classmethod
    def beta_validator(cls, v: float):
        """Validate beta."""
        if not 0 <= v <= 1:
            raise ValueError(
                f"The beta value should be between 0 and 1. Value provided: {v}"
            )
        return v

    @field_validator("pop_ratio")
    @classmethod
    def pop_ratio_validator(cls, v: float):
        """Validate pop_ratio."""
        if not 0 <= v <= 1:
            raise ValueError(
                f"The pop ratio value should be between 0 and 1. Value provided: {v}"
            )
        return v
