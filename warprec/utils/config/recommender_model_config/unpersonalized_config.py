# pylint: disable=duplicate-code
from pydantic import field_validator

from warprec.utils.config.model_configuration import RecomModel
from warprec.utils.config.model_configuration import INT_FIELD
from warprec.utils.config.common import validate_greater_than_zero
from warprec.utils.registry import params_registry


@params_registry.register("Pop")
class Pop(RecomModel):
    """Empty definition of the model Pop."""


@params_registry.register("Random")
class Random(RecomModel):
    """Empty definition of the model Random."""


@params_registry.register("PersonalizedMostPop")

class PersonalizedMostPop(RecomModel):
    """Definition of the model PersonalizedMostPopular."""

    sequence_length: INT_FIELD = 10

    @field_validator("sequence_length")
    @classmethod
    def check_sequence_length(cls, v: list):
        """Validate sequence_length."""
        return validate_greater_than_zero(cls, v, "sequence_length")
