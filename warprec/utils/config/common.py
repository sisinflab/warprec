import difflib
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    Type,
    TypeVar,
    Optional,
    List,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import yaml
from pydantic import BaseModel, ValidationError
from warprec.errors import ConfigurationError, DataNotFoundError, RegistryError
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
        context_labels (Optional[List[str]]): A list of names of the contextual features.
    """

    user_id_label: Optional[str] = "user_id"
    item_id_label: Optional[str] = "item_id"
    rating_label: Optional[str] = "rating"
    timestamp_label: Optional[str] = "timestamp"
    cluster_label: Optional[str] = "cluster"
    context_labels: Optional[List[str]] = []

    @classmethod
    def from_list(cls, labels: List[str]) -> "Labels":
        """Creates a Labels instance from a list of labels.

        Args:
            labels (List[str]): A list of labels in the order of:
                user_id, item_id, rating, timestamp.

        Returns:
            Labels: An instance of the Labels class with the provided labels.

        Raises:
            ConfigurationError: If the input is not a list of length 4.
        """
        if not isinstance(labels, list) | len(labels) != 4:
            raise ConfigurationError("Input must be a list of length 4.")
        return cls(
            user_id_label=labels[0],
            item_id_label=labels[1],
            rating_label=labels[2],
            timestamp_label=labels[3],
        )


def read_configuration_file(path: Union[str, Path], schema: Type[T], kind: str) -> T:
    """Read a YAML configuration file and validate it against its schema.

    The invalid keys pydantic finds are reported together, in one
    ConfigurationError, rather than as pydantic's own error. The checks that
    compare sections run only once every section is valid on its own.

    Args:
        path (Union[str, Path]): The YAML file.
        schema (Type[T]): The pydantic model the file must satisfy.
        kind (str): The kind of configuration, e.g. 'train', for the message.

    Returns:
        T: The validated configuration.

    Raises:
        DataNotFoundError: If the file, or a file it names that must exist when
            it is read (the training data), does not exist.
        ConfigurationError: If the file is not a valid configuration of its kind.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        # Before validating, so that a misspelled required section is named
        # next to the error its absence raises.
        warn_unknown_keys(data, schema)
        return schema.model_validate(data)
    except ValidationError as error:
        message, errors = ConfigurationError.describe_validation(
            error, f"a valid {kind} configuration", path=path
        )
        raise ConfigurationError(message, path=path, errors=errors) from error
    except FileNotFoundError as error:
        raise DataNotFoundError(str(error)) from error


def warn_unknown_keys(data: Any, schema: Type[BaseModel]) -> None:
    """Log every key of a configuration that no section reads.

    Pydantic ignores such a key, so a misspelled one leaves the run on the
    default without a word. Each is logged once, by its dotted path, with the
    closest key its section does read. A section that accepts any key (a
    model's hyperparameters, which the model checks itself) or refuses unknown
    ones (the serving configuration) is left alone, and so are the free-form
    dictionaries handed on as they are (callback kwargs, optimizer, scheduler,
    re-ranker and complex metric params, MLflow tags).

    A schema may hold two class attributes the walk reads: 'section_schemas',
    mapping a field to the type its raw value is walked as when the field is
    declared looser than that, and 'pipeline_sections', mapping every
    top-level section any pipeline reads to its type. One file may serve
    several pipelines, so a section only another pipeline reads is not
    reported, though the keys inside it are checked like any other: a key no
    pipeline reads is a typo whichever pipeline loads the file.

    Args:
        data (Any): The configuration as read from the file.
        schema (Type[BaseModel]): The pydantic model the file must satisfy.
    """
    for key, meant in _unknown_keys(data, schema, ""):
        message = (
            f"The configuration key '{key}' is not one WarpRec reads and is ignored."
        )
        if meant is not None:
            message += f" Did you mean '{meant}'?"
        logger.attention(message)


def _unknown_keys(
    data: Any, schema: Type[BaseModel], path: str
) -> List[Tuple[str, Optional[str]]]:
    """The keys of one section its schema does not read, and their closest.

    Args:
        data (Any): The section as read from the file.
        schema (Type[BaseModel]): The pydantic model of the section.
        path (str): The dotted path of the section, empty at the top.

    Returns:
        List[Tuple[str, Optional[str]]]: Each unknown key's dotted path, with
            the dotted path of the closest key the section reads, or None.
    """
    if not isinstance(data, dict):
        return []

    fields = schema.model_fields
    known = {
        name: info
        for field, info in fields.items()
        for name in {field, info.alias or field}
    }
    overrides = getattr(schema, "section_schemas", {})
    others = getattr(schema, "pipeline_sections", {})
    prefix = f"{path}." if path else ""
    found: List[Tuple[str, Optional[str]]] = []

    for key, value in data.items():
        if not isinstance(key, str):
            continue
        info = known.get(key)
        if info is None and key in others:
            # A section another pipeline reads: checked, not reported.
            found.extend(_unknown_in_value(value, others[key], prefix + key))
            continue
        if info is None:
            if schema.model_config.get("extra") in ("allow", "forbid"):
                continue
            close = difflib.get_close_matches(key, [*known, *others], n=1)
            found.append((prefix + key, prefix + close[0] if close else None))
            continue
        annotation = overrides.get(key, info.annotation)
        found.extend(_unknown_in_value(value, annotation, prefix + key))

    return found


def _unknown_in_value(
    value: Any, annotation: Any, path: str
) -> List[Tuple[str, Optional[str]]]:
    """The unknown keys inside one value, walked as its annotation says.

    Args:
        value (Any): The value as read from the file.
        annotation (Any): The type the value is declared with.
        path (str): The dotted path of the value.

    Returns:
        List[Tuple[str, Optional[str]]]: As _unknown_keys.
    """
    origin = get_origin(annotation)

    if origin in (Union, UnionType):
        # Optional[X] and the like: walk the value as the one alternative of
        # its shape, a section or mapping for a mapping, a list for a list.
        shape = _shape_of_value(value)
        for alternative in get_args(annotation):
            if shape is not None and _shape_of(alternative) is shape:
                return _unknown_in_value(value, alternative, path)
        return []

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _unknown_keys(value, annotation, path)

    args = get_args(annotation)
    if origin is dict and isinstance(value, dict) and len(args) == 2:
        return [
            unknown
            for key, entry in value.items()
            for unknown in _unknown_in_value(entry, args[1], f"{path}.{key}")
        ]

    if origin is list and isinstance(value, list) and args:
        return [
            unknown
            for index, entry in enumerate(value)
            for unknown in _unknown_in_value(entry, args[0], f"{path}[{index}]")
        ]

    return []


def _shape_of(annotation: Any) -> Optional[type]:
    """The container a value of the annotation is read from in a YAML file.

    Args:
        annotation (Any): A type, possibly generic.

    Returns:
        Optional[type]: dict for a section or a mapping, list for a list,
            None for anything else.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return dict
    origin = get_origin(annotation)
    if origin in (dict, list):
        return origin
    return None


def _shape_of_value(value: Any) -> Optional[type]:
    """The container a value read from a YAML file is.

    Args:
        value (Any): The value.

    Returns:
        Optional[type]: dict for a mapping, list for a list, None otherwise.
    """
    if isinstance(value, dict):
        return dict
    if isinstance(value, list):
        return list
    return None


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


def check_seed(value: Any, key: str, draws: str) -> Any:
    """Refuse a seed the random generators it reaches cannot be seeded with.

    A null seed used to reach the generator and fail there, mid-run, and numpy,
    which most draws go through, takes seeds from 0 to 2**32 - 1 only.

    Args:
        value (Any): The configured seed.
        key (str): Where the seed is configured, for the message.
        draws (str): What the seed draws, for the message.

    Returns:
        Any: The seed, unchanged.

    Raises:
        ConfigurationError: If the seed is null or out of range.
    """
    if value is None:
        raise ConfigurationError(
            f"{key} cannot be null: {draws} from it. Leave the key out to use the "
            "default, 42, or give an integer from 0 to 2**32 - 1."
        )
    if isinstance(value, int) and not 0 <= value < 2**32:
        raise ConfigurationError(
            f"{key} must be an integer from 0 to 2**32 - 1, got {value}."
        )
    return value


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
        ConfigurationError: If the allowed list is not a list of strings.
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
            raise ConfigurationError(
                "Validate str list allowed values must be strings."
            )
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
        ConfigurationError: If any of the values are not greater then zero.
    """
    value = _convert_to_list(value)
    value = _scientific_notation_conversion(value)
    for v in value:
        if _check_less_equal_zero(v):
            raise ConfigurationError(
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
        ConfigurationError: If any of the values are not greater or equal then zero.
    """
    value = _convert_to_list(value)
    value = _scientific_notation_conversion(value)
    for v in value:
        if _check_less_than_zero(v):
            raise ConfigurationError(
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
        RegistryError: If any of the values are not a supported similarity.
    """
    value = _convert_to_list(value)
    for v in value:
        if not _check_similarity(v):
            raise RegistryError(
                f"Values of {field} for {cls.__name__} model must be supported similarities. "
                f"Values received as input: {value}. "
                f"Supported similarities: {similarities_registry.list_registered()}",
                registry=similarities_registry.registry_name,
                name=str(v),
                available=similarities_registry.list_registered(),
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
        ConfigurationError: If any of the values are not a supported similarity.
    """
    value = _convert_to_list(value)
    for v in value:
        if not _check_str_values(v, allowed):
            raise ConfigurationError(
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
        ConfigurationError: If any of the values are not a supported profile.
    """
    value = _convert_to_list(value)
    for v in value:
        if not _check_profile(v):
            raise ConfigurationError(
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
        ConfigurationError: If any of the values are not between zero and one.
    """
    value = _convert_to_list(value)
    value = _scientific_notation_conversion(value)
    for v in value:
        if _check_between_zero_and_one(v):
            raise ConfigurationError(
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
        ConfigurationError: If any of the values are not lists of layers.
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
                raise ConfigurationError(
                    f"Values of {field} for {cls.__name__} model must be positive layer values. "
                    f"Values received as input: {value}"
                )
    if strat:
        value.insert(0, strat)
    return value


def validate_value_list(cls: Type[T], value: Any, field: str) -> list:
    """Validate a hyperparameter whose value is itself a list.

    A modality list or a weight list is one value, not a set of alternatives, so
    it is wrapped the way a layer list is before the search space sees it.

    Args:
        cls (Type[T]): Class type of original Pydantic BaseModel.
        value (Any): A value or a list of values to be validated.
        field (str): The name of the field to be validated.

    Returns:
        list: A list of validated values in the correct format.

    Raises:
        ConfigurationError: If any of the values is an empty list.
    """
    if not isinstance(value, list):
        value = [value]
    if not isinstance(value[-1], list):
        value = [value]

    strat = value.pop(0) if isinstance(value[0], str) else None

    for entry in value:
        if not entry:
            raise ConfigurationError(
                f"Values of {field} for {cls.__name__} model must name at least "
                f"one entry. Values received as input: {value}"
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


def validate_numeric_values(value: Any) -> list:
    """Validate a hyperparameter.

    Valid values must be numbers.

    Args:
        value (Any): A value or a list of values to be validated.

    Returns:
        list: A list of validated values in the correct format.
    """
    value = _convert_to_list(value)
    value = _scientific_notation_conversion(value)
    return value
