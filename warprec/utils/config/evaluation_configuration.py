from typing import List, Literal, Optional, Dict, Any

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


class PropensityConfig(BaseModel):
    """Definition of the propensity model the debiased estimators read.

    Attributes:
        estimator (Optional[Literal["uniform", "popularity"]]): How the probability
            that an item was observed is estimated. 'popularity' reads it off the
            training interaction counts, 'uniform' applies no correction.
        power (Optional[float]): The exponent applied to the normalised counts.
            Lower values flatten the correction, 0 removes it.
        clip (Optional[float]): The smallest propensity any item may carry. The
            estimators divide by it, so the floor bounds their variance.
    """

    estimator: Optional[Literal["uniform", "popularity"]] = "uniform"
    power: Optional[float] = 0.5
    clip: Optional[float] = 0.1

    @field_validator("power")
    @classmethod
    def check_power(cls, v: float):
        """Validate the exponent."""
        if v < 0:
            raise ValueError(f"Propensity power must not be negative, got {v}.")
        return v

    @field_validator("clip")
    @classmethod
    def check_clip(cls, v: float):
        """Validate the propensity floor."""
        if not 0 < v <= 1:
            raise ValueError(f"Propensity clip must lie in (0, 1], got {v}.")
        return v


class ComplexMetricConfig(BaseModel):
    """Definition of a metric that carries its own parameters.

    Attributes:
        name (str): The registered name of the metric.
        params (Dict[str, Any]): The parameters passed to the metric on construction.
    """

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str):
        """Validate the metric name against the registry."""
        if v.upper() not in metric_registry.list_registered():
            raise ValueError(f"Metric {v} not found in registry.")
        return v


class EvaluationConfig(BaseModel):
    """Definition of Evaluation configuration.

    Attributes:
        top_k (List[int]): List of cutoffs to evaluate.
        metrics (List[str]): List of metrics to compute during evaluation.
        complex_metrics (List[ComplexMetricConfig]): List of metrics
            which requires further parameters to be instantiated.
        propensity (PropensityConfig): The propensity model the debiased estimators
            read. Defaults to applying no correction.
        validation_metric (Optional[str]): The metric/loss that will
            validate each trial in Ray Tune.
        batch_size (Optional[int]): Batch size used during evaluation.
        strategy (Optional[str]): Evaluation strategy, either "full" or "sampled".
        num_negatives (Optional[int]): Number of negative samples to use in "sampled" strategy.
        candidates (Optional[Literal["all", "cold", "warm"]]): Which items a run is
            allowed to rank. 'cold' keeps only the items with no training
            interaction and 'warm' only the rest, which is what makes a cold-start
            protocol measure cold-start rather than the warm catalogue around it.
            Defaults to 'all'. Item-side only, and ignored by sampled evaluation.
        mask_seen (Optional[Literal["auto", "context", "pair", "none"]]): Which
            already-seen items are excluded from the ranking. 'auto' excludes items
            seen in the same context when the dataset has contextual columns and
            every seen item otherwise, 'context' and 'pair' force either behaviour,
            'none' excludes nothing. Defaults to 'auto'.
        seed (Optional[int]): Random seed for reproducibility. Used in negative sampling.
        stat_significance (Optional[StatSignificance]): Statistical significance configuration.
        full_evaluation_on_report (Optional[bool]): Wether or not to compute all metric
            for each report produced.
        max_metric_per_row (Optional[int]): Number of metrics to show in each row on console.
        save_evaluation (Optional[bool]): Wether or not to save the evaluation.
        save_per_user (Optional[bool]): Wether or not to save per-user evaluation.
    """

    top_k: List[int]
    metrics: List[str]
    complex_metrics: List[ComplexMetricConfig] = Field(default_factory=list)
    propensity: PropensityConfig = Field(default_factory=PropensityConfig)
    validation_metric: Optional[str] = "nDCG@10"
    batch_size: Optional[int] = 1024
    strategy: Optional[str] = "full"  # or "sampled"
    num_negatives: Optional[int] = 99
    candidates: Optional[Literal["all", "cold", "warm"]] = "all"
    mask_seen: Optional[Literal["auto", "context", "pair", "none"]] = "auto"
    seed: Optional[int] = 42
    stat_significance: Optional[StatSignificance] = Field(
        default_factory=StatSignificance
    )
    full_evaluation_on_report: Optional[bool] = False
    max_metric_per_row: Optional[int] = 4
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

        _, top_k = v.split("@")
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
