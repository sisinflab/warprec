from typing import Any, Type, TypeVar, Optional, List

from pydantic import BaseModel
from warprec.utils.enums import SearchSpace
from warprec.utils.registry import similarities_registry
from warprec.utils.logger import logger

T = TypeVar("T", bound=BaseModel)


class Labels(BaseModel):
    """Definition of the label sub-configuration.

    This class reads and optionally overrides the default labels of important data.

    Attributes:
        user_id_label (Optional[str]): Name of the user ID label. Defaults to 'user_id'.
        item_id_label (Optional[str]): Name of the item ID label. Defaults to 'item_id'.
        rating_label (Optional[str]): Name of the rating label. Defaults to 'rating'.
        timestamp_label (Optional[str]): Name of the timestamp label. Defaults to 'timestamp'.
        cluster_label (Optional[str]): Name of the cluster label. Defaults to 'cluster'.
    """

    user_id_label: Optional[str] = "user_id"
    item_id_label: Optional[str] = "item_id"
    rating_label: Optional[str] = "rating"
    timestamp_label: Optional[str] = "timestamp"
    cluster_label: Optional[str] = "cluster"

    @classmethod
    def from_list(cls, labels: List[str]) -> "Labels":
        """Creates a Labels instance from a list of labels.

        Args:
            labels (List[str]): A list of labels in the order of:
                user_id, item_id, rating, timestamp.

        Returns:
            Labels: An instance of the Labels class with the provided labels.

        Raises:
            ValueError: If the input is not a list of length 4.
        """
        if not isinstance(labels, list) | len(labels) != 4:
            raise ValueError("Input must be a list of length 4.")
        return cls(
            user_id_label=labels[0],
            item_id_label=labels[1],
            rating_label=labels[2],
            timestamp_label=labels[3],
        )


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


def _check_str_values(value: Any, allowed: list) -> bool:
    """Check if the field is in allowed list.

    Args:
        value (Any): The value to check.
        allowed (list): List of allowed values.

    Returns:
        bool: True if the value is in the allowed list, False otherwise.

    Raises:
        ValueError: If the allowed list is not a list of strings.
    """

    # Check if not string
    if not isinstance(value, str):
        return False

    # Check for search space value
    if value.lower() in {SearchSpace.CHOICE.value, SearchSpace.GRID.value}:
        return True

    # Cast to lower all the string in the allowed list
    lower_allowed = []
    for v in allowed:
        if not isinstance(v, str):
            raise ValueError("Validate str list allowed values must be strings.")
        lower_allowed.append(v.lower())
    allowed = lower_allowed

    # Check if supported similarity
    return value.lower() in allowed


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


def _scientific_notation_conversion(values: list) -> list:
    """Converts an input list to force scientific
    notation numbers to be floats.

    Args:
        values (list): The original list.

    Returns:
        list: The converted list.
    """

    def _attempt_conversion(value: Any) -> Any:
        """Performs the conversion logic for a single element.

        Args:
            value (Any): The original value.

        Returns:
            Any: The converted value.
        """
        # Skip non-string elements immediately (int, float, etc.)
        if not isinstance(value, str):
            return value

        # Skip strings that don't contain 'e' or 'E' (e.g., "grid", "randint")
        if "e" not in value.lower():
            return value

        # Attempt the safe conversion for scientific notation
        try:
            # Successful conversion to float (e.g., "1e10")
            return float(value)
        except ValueError:
            # We return the original string in this case.
            return value

    # Apply conversion logic to the entire list
    return [_attempt_conversion(value) for value in values]


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
    value = _scientific_notation_conversion(value)
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
    value = _scientific_notation_conversion(value)
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


def validate_str_list(cls: Type[T], value: Any, allowed: list, field: str) -> list:
    """Validate a hyperparameter.

    Valid values must be supported similarity.

    Args:
        cls (Type[T]): Class type of original Pydantic BaseModel.
        value (Any): A value or a list of values to be validated.
        allowed (list): List of allowed values.
        field (str): The name of the field to be validated.

    Returns:
        list: A list of validated values in the correct format.

    Raises:
        ValueError: If any of the values are not a supported similarity.
    """
    value = _convert_to_list(value)
    for v in value:
        if not _check_str_values(v, allowed):
            raise ValueError(
                f"Values of {field} for {cls.__name__} model are not allowed. "
                f"Values received as input: {value}. "
                f"Supported values: {allowed}"
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
    value = _scientific_notation_conversion(value)
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
