from pydantic import BaseModel, field_validator


class EstimateConfig(BaseModel):
    """Definition of the estimate-specific configuration section.

    Attributes:
        train_batches (int): Number of measured training batches to estimate.
        eval_batches (int): Number of measured evaluation batches to estimate.
        warmup_batches (int): Number of warmup batches to skip from measurements.
    """

    train_batches: int = 100
    eval_batches: int = 100
    warmup_batches: int = 10

    @field_validator("train_batches", "eval_batches")
    @classmethod
    def check_positive_batches(cls, v: int) -> int:
        """Validates that measured batch counts are strictly positive."""
        if v <= 0:
            raise ValueError(
                "Measured estimate batch counts must be greater than zero."
            )
        return v

    @field_validator("warmup_batches")
    @classmethod
    def check_non_negative_warmup(cls, v: int) -> int:
        """Validates that warmup batches are non-negative."""
        if v < 0:
            raise ValueError("Warmup batches must be greater than or equal to zero.")
        return v
