# pylint: disable=duplicate-code
from pydantic import field_validator
from warprec.utils.config.model_configuration import (
    RecomModel,
    INT_FIELD,
    STR_FIELD,
    FLOAT_FIELD,
)
from warprec.utils.config.common import (
    validate_greater_than_zero,
    validate_similarity,
)
from warprec.utils.registry import params_registry


@params_registry.register("ItemKNN")
class ItemKNN(RecomModel):
    """Definition of the model ItemKNN.

    Attributes:
        k (INT_FIELD): List of values for neighbor.
        similarity (STR_FIELD): List of names of similarity functions.
    """

    k: INT_FIELD
    similarity: STR_FIELD

    @field_validator("k")
    @classmethod
    def check_k(cls, v: list):
        """Validate k."""
        return validate_greater_than_zero(cls, v, "k")

    @field_validator("similarity")
    @classmethod
    def check_similarity(cls, v: list):
        """Validate similarity."""
        return validate_similarity(cls, v, "similarity")


@params_registry.register("ItemKNN-TD")
class ItemKNNTD(RecomModel):
    """Definition of the model ItemKNN-TD.

    Attributes:
        k (INT_FIELD): List of values for neighbor.
        similarity (STR_FIELD): List of names of similarity functions.
        beta (FLOAT_FIELD): List of values for decay rate parameter.
    """

    k: INT_FIELD
    similarity: STR_FIELD
    beta: FLOAT_FIELD

    @field_validator("k")
    @classmethod
    def check_k(cls, v: list):
        """Validate k."""
        return validate_greater_than_zero(cls, v, "k")

    @field_validator("similarity")
    @classmethod
    def check_similarity(cls, v: list):
        """Validate similarity."""
        return validate_similarity(cls, v, "similarity")

    @field_validator("beta")
    @classmethod
    def check_beta(cls, v: list):
        """Validate beta."""
        return validate_greater_than_zero(cls, v, "beta")


@params_registry.register("UserKNN")
class UserKNN(RecomModel):
    """Definition of the model UserKNN.

    Attributes:
        k (INT_FIELD): List of values for neighbor.
        similarity (STR_FIELD): List of names of similarity functions.
    """

    k: INT_FIELD
    similarity: STR_FIELD

    @field_validator("k")
    @classmethod
    def check_k(cls, v: list):
        """Validate k."""
        return validate_greater_than_zero(cls, v, "k")

    @field_validator("similarity")
    @classmethod
    def check_similarity(cls, v: list):
        """Validate similarity."""
        return validate_similarity(cls, v, "similarity")


@params_registry.register("UserKNN-TD")
class UserKNNTD(RecomModel):
    """Definition of the model UserKNN-TD.

    Attributes:
        k (INT_FIELD): List of values for neighbor.
        similarity (STR_FIELD): List of names of similarity functions.
        beta (FLOAT_FIELD): List of values for decay rate parameter.
    """

    k: INT_FIELD
    similarity: STR_FIELD
    beta: FLOAT_FIELD

    @field_validator("k")
    @classmethod
    def check_k(cls, v: list):
        """Validate k."""
        return validate_greater_than_zero(cls, v, "k")

    @field_validator("similarity")
    @classmethod
    def check_similarity(cls, v: list):
        """Validate similarity."""
        return validate_similarity(cls, v, "similarity")

    @field_validator("beta")
    @classmethod
    def check_beta(cls, v: list):
        """Validate beta."""
        return validate_greater_than_zero(cls, v, "beta")
