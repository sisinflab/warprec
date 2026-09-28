from typing import Literal, Optional

from pydantic import BaseModel, field_validator

NegativeSampling = Literal["uniform", "popularity"]
SequencePooling = Literal["mean", "sum", "max"]


class TrainingConfig(BaseModel):
    """Definition of the training configuration part of the configuration file.

    These options shape the signal a model learns from rather than the data the
    reader produces: the dataset is identical whichever value they take, only
    the examples drawn from it differ. That is why they live here and not under
    the reader, where they were first shipped.

    Attributes:
        negative_sampling (Optional[NegativeSampling]): How negatives are drawn during
            training. 'uniform' gives every item the same chance, 'popularity' draws
            proportionally to a dampened interaction count. Defaults to 'uniform'.
        neg_alpha (Optional[float]): The exponent 'popularity' applies
            to the interaction counts before drawing. Defaults to 0.75, the usual
            choice, which keeps the head likely without letting it dominate. Zero
            makes every item equally likely, recovering 'uniform'; one draws in
            exact proportion to the counts. Ignored by 'uniform'.
        sequence_pooling (Optional[SequencePooling]): How the values of a multi-valued
            field are combined into the single vector the field contributes. Defaults
            to 'mean', which matches the normalised multi-hot encoding the
            factorisation-machine literature defines these models over.
    """

    negative_sampling: Optional[NegativeSampling] = "uniform"
    neg_alpha: Optional[float] = 0.75
    sequence_pooling: Optional[SequencePooling] = "mean"

    @field_validator("neg_alpha")
    @classmethod
    def alpha_is_not_negative(cls, v: Optional[float]) -> Optional[float]:
        """Reject an exponent that would invert the distribution.

        Args:
            v (Optional[float]): The configured value.

        Returns:
            Optional[float]: The value, unchanged.

        Raises:
            ValueError: If the exponent is negative.
        """
        if v is not None and v < 0:
            raise ValueError(
                f"neg_alpha must not be negative, got {v}. A negative "
                "exponent makes the rarest items the most likely negatives, which is "
                "not what 'popularity' sampling means."
            )
        return v
