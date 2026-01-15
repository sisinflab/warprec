# pylint: disable=duplicate-code
from typing import ClassVar

from pydantic import field_validator
from warprec.utils.config.model_configuration import (
    RecomModel,
    STR_FIELD,
)
from warprec.utils.config.common import validate_similarity, validate_profile
from warprec.utils.registry import params_registry


@params_registry.register("VSM")
class VSM(RecomModel):
    """Definition of the model VSM.

    Attributes:
        similarity (STR_FIELD): List of names of similarity functions.
        user_profile (STR_FIELD): List of user profile computations.
        item_profile (STR_FIELD): List of item profile computations.
        need_side_information (ClassVar[bool]): Wether or not the model needs side information.
    """

    similarity: STR_FIELD
    user_profile: STR_FIELD
    item_profile: STR_FIELD
    need_side_information: ClassVar[bool] = True

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

    @field_validator("item_profile")
    @classmethod
    def check_item_profile(cls, v: list):
        """Validate item_profile."""
        return validate_profile(cls, v, "item_profile")
