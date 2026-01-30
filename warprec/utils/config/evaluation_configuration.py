from typing import List, Optional

import re
from pydantic import BaseModel, field_validator, Field
from warprec.utils.registry import metric_registry
from warprec.utils.logger import logger


class Corrections(BaseModel):
    """Definition of corrections configuration.

    Attributes:
        bonferroni (Optional[bool]): Whether to apply Bonferroni correction.
        holm_bonferroni (Optional[bool]): Whether to apply Holm-Bonferroni correction.
        fdr (Optional[bool]): Whether to apply False Discovery Rate correction.
        alpha (Optional[float]): Significance level for statistical tests.
    """

    bonferroni: Optional[bool] = False
    holm_bonferroni: Optional[bool] = False
    fdr: Optional[bool] = False
    alpha: Optional[float] = 0.05

    def requires_correction(self) -> bool:
        """Check if any correction is enabled.

        Returns:
            bool: True if any correction is enabled, False otherwise.
        """
        return any(self.model_dump(exclude=["alpha"]).values())  # type: ignore[arg-type]


class StatSignificance(BaseModel):
    """Definition of statistical significance configuration.

    Attributes:
        paired_t_test (Optional[bool]): Whether to perform paired t-test.
        wilcoxon_test (Optional[bool]): Whether to perform Wilcoxon test.
        kruskal_test (Optional[bool]): Whether to perform Kruskal-Wallis test.
        whitney_u_test (Optional[bool]): Whether to perform Mann-Whitney U test.
        corrections (Optional[Corrections]): Corrections configuration for statistical tests.
    """

    paired_t_test: Optional[bool] = False
    wilcoxon_test: Optional[bool] = False
    kruskal_test: Optional[bool] = False
    whitney_u_test: Optional[bool] = False
    corrections: Optional[Corrections] = Field(default_factory=Corrections)

    def requires_stat_significance(self) -> bool:
        """Check if any statistical significance test is enabled.

        Returns:
            bool: True if any test is enabled, False otherwise.
        """
        return any(self.model_dump(exclude=["corrections"]).values())  # type: ignore[arg-type]


class EvaluationConfig(BaseModel):
    """Definition of Evaluation configuration.

    Attributes:
        top_k (List[int]): List of cutoffs to evaluate.
        metrics (List[str]): List of metrics to compute during evaluation.
        validation_metric (Optional[str]): The metric/loss that will
            validate each trial in Ray Tune.
        batch_size (Optional[int]): Batch size used during evaluation.
        strategy (Optional[str]): Evaluation strategy, either "full" or "sampled".
        num_negatives (Optional[int]): Number of negative samples to use in "sampled" strategy.
        seed (Optional[int]): Random seed for reproducibility. Used in negative sampling.
        stat_significance (Optional[StatSignificance]): Statistical significance configuration.
        full_evaluation_on_report (Optional[bool]): Wether or not to compute all metric
            for each report produced.
        max_metric_per_row (Optional[int]): Number of metrics to show in each row on console.
        beta (Optional[float]): The beta value used in some metrics like F1 score.
        pop_ratio (Optional[float]): The percentage of item transactions that
            will be considered popular.
        save_evaluation (Optional[bool]): Wether or not to save the evaluation.
        save_per_user (Optional[bool]): Wether or not to save per-user evaluation.
    """

    top_k: List[int]
    metrics: List[str]
    validation_metric: Optional[str] = "nDCG@5"
    batch_size: Optional[int] = 1024
    strategy: Optional[str] = "full"  # or "sampled"
    num_negatives: Optional[int] = 99
    seed: Optional[int] = 42
    stat_significance: Optional[StatSignificance] = Field(
        default_factory=StatSignificance
    )
    full_evaluation_on_report: Optional[bool] = False
    max_metric_per_row: Optional[int] = 4
    beta: Optional[float] = 1.0
    pop_ratio: Optional[float] = 0.8
    save_evaluation: Optional[bool] = True
    save_per_user: Optional[bool] = False

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
        supported_relevance = ["binary", "discounted"]

        for metric in v:
            # Check for F1-extended metric
            match_f1 = re.match(
                r"F1\[\s*(.*?)\s*,\s*(.*?)\s*\]", metric
            )  # Expected syntax: F1[metric_1, metric_2]

            # Check for EFD or EPC extended metric
            match_efd_epc = re.match(
                r"(EFD|EPC)\[\s*(.*?)\s*\]", metric
            )  # Expected syntax: EFD[value] or EPC[value]

            # Check for Hypervolume metric
            match_hv = re.match(
                r"Hypervolume\s*\[(.*?)\]\s*\((.*?)\)\s*\((.*?)\)", metric
            )

            match_ed = re.match(
                r"EucDistance\s*\[(.*?)\]\s*\((.*?)\)\s*\((.*?)\)", metric
            )

            match_var = re.match(
                r"Variance\[\s*(.*?)\s*\]", metric
            )

            if match_f1:
                # Extract sub metrics
                metric_1 = match_f1.group(1)
                metric_2 = match_f1.group(2)

                # Check if sub metrics exists inside registry
                if metric_1.upper() not in metric_registry.list_registered():
                    raise ValueError(
                        f"In {metric} the sub metric {metric_1} not in metric registry. This is the list"
                        f"of supported metrics: {metric_registry.list_registered()}"
                    )

                if metric_2.upper() not in metric_registry.list_registered():
                    raise ValueError(
                        f"In {metric} the sub metric {metric_2} not in metric registry. This is the list"
                        f"of supported metrics: {metric_registry.list_registered()}"
                    )

                continue  # Skip normal metric check

            if match_hv or match_ed:
                # Extract sub metrics list
                try:
                    metric_list = [m.strip() for m in match_hv.group(1).split(",")]
                except AttributeError:
                    metric_list = [m.strip() for m in match_ed.group(1).split(",")]

                # Check if sub metrics exists inside registry
                for ml in metric_list:
                    if ml.upper() not in metric_registry.list_registered():
                        raise ValueError(
                            f"In {metric} the sub metric {ml} not in metric registry. This is the list"
                            f"of supported metrics: {metric_registry.list_registered()}"
                        )

                continue

            if match_efd_epc:
                relevance = match_efd_epc.group(2)

                if relevance.lower() not in supported_relevance:
                    raise ValueError(
                        f"In {metric} the relevance score {relevance} is not supported. "
                        f"These are the supported relevance scores: {supported_relevance}."
                    )

                continue  # Skip normal metric check

            if match_var:
                # Extract sub metrics
                metric_1 = match_var.group(1)

                # Check if sub metrics exists inside registry
                if metric_1.upper() not in metric_registry.list_registered():
                    raise ValueError(
                        f"In {metric} the sub metric {metric_1} not in metric registry. This is the list"
                        f"of supported metrics: {metric_registry.list_registered()}"
                    )

                continue  # Skip normal metric check

            # Check for normal metrics
            if metric.upper() not in metric_registry.list_registered():
                raise ValueError(
                    f"Metric {metric} not in metric registry. This is the list"
                    f"of supported metrics: {metric_registry.list_registered()}"
                )

        # Check for duplicates
        len_before = len(v)
        v = cls.check_duplicates_in_list(v)
        len_after = len(v)

        if len_after < len_before:
            logger.attention(
                f"Duplicated metrics found inside evaluation. "
                f"{len_before - len_after} metrics removed."
            )
        return v

    @field_validator("validation_metric")
    @classmethod
    def check_validation_metric(cls, v: str):
        """Validate validation metric."""
        if v is None:
            raise ValueError("Validation metric must be provided.")
        if "@" not in v:
            raise ValueError(
                f"Validation metric {v} not valid. Validation metric "
                f"should be defined as: metric_name@top_k."
            )
        if v.count("@") > 1:
            raise ValueError(
                "Validation metric contains more than one @, check your configuration file."
            )
        metric, top_k = v.split("@")
        if metric.upper() not in metric_registry.list_registered():
            raise ValueError(
                f"Metric {metric} not in metric registry. This is the list"
                f"of supported metrics: {metric_registry.list_registered()}"
            )
        if not top_k.isnumeric():
            raise ValueError(
                "Validation metric should be provided with a top_k number."
            )
        return v

    @field_validator("strategy")
    @classmethod
    def strategy_validator(cls, v: str):
        """Validate strategy."""
        if v not in ["full", "sampled"]:
            raise ValueError(
                f"The strategy value should be either 'full' or 'sampled'. Value provided: {v}"
            )
        return v

    @field_validator("num_negatives")
    @classmethod
    def num_negatives_validator(cls, v: int):
        """Validate num_negatives."""
        if v is not None and v <= 0:
            raise ValueError(
                f"The num_negatives value should be a positive integer. Value provided: {v}"
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

    @classmethod
    def check_duplicates_in_list(cls, values: list) -> list:
        """Check and remove duplicate elements from list
        preserving the order.

        Args:
            values (list): The original list.

        Returns:
            list: The list without duplicates.
        """
        list_without_duplicates = []
        unique_elements = set()

        for v in values:
            if v not in unique_elements:
                list_without_duplicates.append(v)
                unique_elements.add(v)
        return list_without_duplicates
