from typing import ClassVar

from pydantic import field_validator
from warprec.utils.config.model_configuration import (
    RecomModel,
    INT_FIELD,
    STR_FIELD,
)
from warprec.utils.config.common import (
    validate_greater_than_zero,
    validate_similarity,
    validate_profile,
)
from warprec.utils.registry import params_registry


@params_registry.register("AttributeItemKNN")
class AttributeItemKNN(RecomModel):
    """Definition of the model AttributeItemKNN.

    Attributes:
        k (INT_FIELD): List of values for neighbor.
        similarity (STR_FIELD): List of names of similarity functions.
        need_side_information (ClassVar[bool]): Wether or not the model needs side information.
    """

    k: INT_FIELD
    similarity: STR_FIELD
    need_side_information: ClassVar[bool] = True

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


@params_registry.register("AttributeUserKNN")
class AttributeUserKNN(RecomModel):
    """Definition of the model AttributeUserKNN.

    Attributes:
        k (INT_FIELD): List of values for neighbor.
        similarity (STR_FIELD): List of names of similarity functions.
        user_profile (STR_FIELD): List of user profile computations.
        need_side_information (ClassVar[bool]): Wether or not the model needs side information.
    """

    k: INT_FIELD
    similarity: STR_FIELD
    user_profile: STR_FIELD
    need_side_information: ClassVar[bool] = True

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

    @field_validator("user_profile")
    @classmethod
    def check_user_profile(cls, v: list):
        """Validate user_profile."""
        return validate_profile(cls, v, "user_profile")


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
