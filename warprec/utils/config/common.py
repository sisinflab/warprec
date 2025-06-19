from typing import Any, Type, TypeVar

from pydantic import BaseModel
from warprec.utils.enums import SearchSpace
from warprec.utils.registry import similarities_registry
from warprec.utils.logger import logger

T = TypeVar("T", bound=BaseModel)


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


def _convert_to_list(value: Any) -> list:
    """Convert the input value to a list if it is not already a list.

    Args:
        value (Any): The value to convert.

    Returns:
        list: The converted list.
    """
    if isinstance(value, list):
        return value
    return [value]


def _check_less_equal_zero(value: Any) -> bool:
    """Check if the field is numerical and less than or equal to zero.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is less than or equal to zero, False otherwise.
    """

    return isinstance(value, (float, int)) and value <= 0


def _check_less_than_zero(value: Any) -> bool:
    """Check if the field is numerical and less than zero.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is less than zero, False otherwise.
    """

    return isinstance(value, (float, int)) and value < 0


def _check_zero_to_one(value: Any) -> bool:
    """Check if the field is numerical and between 0 and 1.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is between 0 and 1, False otherwise.
    """

    return isinstance(value, (float, int)) and 0 <= value <= 1


def _check_similarity(value: Any) -> bool:
    """Check if the field is correct string.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a correct similarity, False otherwise.
    """

    # Check if not string
    if not isinstance(value, str):
        return False

    # Check for search space value
    if value.lower() in {SearchSpace.CHOICE.value, SearchSpace.GRID.value}:
        return True

    # Check if supported similarity
    return value.upper() in similarities_registry.list_registered()


def _check_profile(value: Any) -> bool:
    """Check if the field is correct string.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is a correct user profile, False otherwise.
    """

    return isinstance(value, str) and (
        value.lower() == SearchSpace.CHOICE.value
        or value.lower() == SearchSpace.GRID.value
        or value.lower() in ["binary", "tfidf"]
    )


def _check_between_zero_and_one(value: Any) -> bool:
    """Check if the field is correct value between 0 and 1.

    Args:
        value (Any): The value to check.

    Returns:
        bool: True if the value is between 0 and 1.
    """
    return isinstance(value, (float, int)) and (value < 0 or value > 1)


def validate_greater_than_zero(cls: Type[T], value: Any, field: str) -> list:
    """Validate a hyperparameter.

    Valid values must be greater than zero.

    Args:
        cls (Type[T]): Class type of original Pydantic BaseModel.
        value (Any): A value or a list of values to be validated.
        field (str): The name of the field to be validated.

    Returns:
        list: A list of validated values in the correct format.

    Raises:
        ValueError: If any of the values are not greater then zero.
    """
    value = _convert_to_list(value)
    for v in value:
        if _check_less_equal_zero(v):
            raise ValueError(
                f"Values of {field} for {cls.__name__} model must be > 0. "
                f"Values received as input: {value}"
            )
    return value


def validate_greater_equal_than_zero(cls: Type[T], value: Any, field: str) -> list:
    """Validate a hyperparameter.

    Valid values must be greater or equal than zero.

    Args:
        cls (Type[T]): Class type of original Pydantic BaseModel.
        value (Any): A value or a list of values to be validated.
        field (str): The name of the field to be validated.

    Returns:
        list: A list of validated values in the correct format.

    Raises:
        ValueError: If any of the values are not greater or equal then zero.
    """
    value = _convert_to_list(value)
    for v in value:
        if _check_less_than_zero(v):
            raise ValueError(
                f"Values of {field} for {cls.__name__} model must be >= 0. "
                f"Values received as input: {value}"
            )
    return value


def validate_similarity(cls: Type[T], value: Any, field: str) -> list:
    """Validate a hyperparameter.

    Valid values must be supported similarity.

    Args:
        cls (Type[T]): Class type of original Pydantic BaseModel.
        value (Any): A value or a list of values to be validated.
        field (str): The name of the field to be validated.

    Returns:
        list: A list of validated values in the correct format.

    Raises:
        ValueError: If any of the values are not a supported similarity.
    """
    value = _convert_to_list(value)
    for v in value:
        if not _check_similarity(v):
            raise ValueError(
                f"Values of {field} for {cls.__name__} model must be supported similarities. "
                f"Values received as input: {value}. "
                f"Supported similarities: {similarities_registry.list_registered()}"
            )
    return value


def validate_profile(cls: Type[T], value: Any, field: str) -> list:
    """Validate a hyperparameter.

    Valid values must be supported profile.

    Args:
        cls (Type[T]): Class type of original Pydantic BaseModel.
        value (Any): A value or a list of values to be validated.
        field (str): The name of the field to be validated.

    Returns:
        list: A list of validated values in the correct format.

    Raises:
        ValueError: If any of the values are not a supported profile.
    """
    value = _convert_to_list(value)
    for v in value:
        if not _check_profile(v):
            raise ValueError(
                f"Values of {field} for {cls.__name__} model must be 'binary' or 'tfidf'. "
                f"Values received as input: {value}. "
            )
    return value


def validate_between_zero_and_one(cls: Type[T], value: Any, field: str) -> list:
    """Validate a hyperparameter.

    Valid values must be between zero and one.

    Args:
        cls (Type[T]): Class type of original Pydantic BaseModel.
        value (Any): A value or a list of values to be validated.
        field (str): The name of the field to be validated.

    Returns:
        list: A list of validated values in the correct format.

    Raises:
        ValueError: If any of the values are not between zero and one.
    """
    value = _convert_to_list(value)
    for v in value:
        if _check_between_zero_and_one(v):
            raise ValueError(
                f"Values of {field} for {cls.__name__} model must be >= 0 and <= 1. "
                f"Values received as input: {value}"
            )
    return value


def validate_layer_list(cls: Type[T], value: Any, field: str) -> list:
    """Validate a hyperparameter.

    Valid values must be lists of layers.

    Args:
        cls (Type[T]): Class type of original Pydantic BaseModel.
        value (Any): A value or a list of values to be validated.
        field (str): The name of the field to be validated.

    Returns:
        list: A list of validated values in the correct format.

    Raises:
        ValueError: If any of the values are not lists of layers.
    """
    strat = None  # Init strategy
    if not isinstance(value, list):
        value = [value]
    if not isinstance(value[-1], list):
        value = [value]
    if isinstance(value[0], str):
        strat = value.pop(0)
    for layer in value:
        for v in layer:
            if v <= 0:
                raise ValueError(
                    f"Values of {field} for {cls.__name__} model must be positive layer values. "
                    f"Values received as input: {value}"
                )
    if strat:
        value.insert(0, strat)
    return value


def validate_bool_values(value: Any) -> list:
    """Validate a hyperparameter.

    Valid values must be boolean.

    Args:
        value (Any): A value or a list of values to be validated.

    Returns:
        list: A list of validated values in the correct format.
    """
    return _convert_to_list(value)
