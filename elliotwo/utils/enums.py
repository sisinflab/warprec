from enum import Enum


class RatingType(str, Enum):
    """Represents the types of rating supported.

    This enum is used to track the possible rating definition:
        - EXPLICIT: The rating can be in a range, usually using floats. \
            The scores will be read from raw transaction data.
        - IMPLICIT: The rating will all be set to 1. \
            These scores won't be read from raw transaction data.
    """

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class SplittingStrategies(str, Enum):
    """Represents the types of splitting strategies supported.

    This enum is used to track the possible splitting strategies:
        - NONE: The splitting will not be performed.
        - RANDOM: The splitting will be random. \
            A seed will bi used to ensure reproducibility.
        - LEAVE_ONE_OUT: The splitting will remove just one element. \
            The elemente chosen will be the same if a seed is set.
        - TEMPORAL: The splitting will be based on the timestamp. \
            Timestamps will be mandatory if this strategy is chosen.
    """

    NONE = "none"
    RANDOM = "random"
    LEAVE_ONE_OUT = "leave-one-out"
    TEMPORAL = "temporal"
