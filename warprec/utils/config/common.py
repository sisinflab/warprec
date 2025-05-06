from typing import Any

from warprec.utils.enums import SearchSpace
from warprec.utils.registry import similarities_registry
from warprec.utils.logger import logger


def check_separator(sep: str) -> str:
    """This method checks the separator, if it's not in a correct format
        then it is set to the default separator.

    Args:
        sep (str): The separator to check.

    Returns:
        str: The validated separator.
    """
    try:
        sep = sep.encode().decode("unicode_escape")
    except UnicodeDecodeError:
        logger.negative(
            f"The string {sep} is not a valid separator. Using default separator {'\t'}."
        )
        sep = "\t"
    return sep


def convert_to_list(value: Any) -> list:
    """Convert the input value to a list if it is not already a list.

    Args:
        value (Any): The value to convert.

    Returns:
        list: The converted list.
    """
    if isinstance(value, list):
        return value
    return [value]


def check_less_equal_zero(value: Any) -> bool:
    """Check if the field is numerical and less than or equal to zero.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is less than or equal to zero, False otherwise.
    """

    return isinstance(value, (float, int)) and value <= 0


def check_less_than_zero(value: Any) -> bool:
    """Check if the field is numerical and less than zero.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is less than zero, False otherwise.
    """

    return isinstance(value, (float, int)) and value < 0


def check_zero_to_one(value: Any) -> bool:
    """Check if the field is numerical and between 0 and 1.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is between 0 and 1, False otherwise.
    """

    return isinstance(value, (float, int)) and 0 <= value <= 1


def check_similarity(value: Any) -> bool:
    """Check if the field is correct string.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a correct similarity, False otherwise.
    """

    return (
        isinstance(value, str)
        and value.lower() != SearchSpace.CHOICE.value
        and value.lower() != SearchSpace.GRID.value
        and value.upper() in similarities_registry.list_registered()
    )


def check_user_profile(value: Any) -> bool:
    """Check if the field is correct string.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a correct user profile, False otherwise.
    """

    return (
        isinstance(value, str)
        and value.lower() != SearchSpace.CHOICE.value
        and value.lower() != SearchSpace.GRID.value
        and value.upper() in ["binary", "tfidf"]
    )
