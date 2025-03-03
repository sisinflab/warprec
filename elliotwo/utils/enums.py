from enum import Enum


class RatingType(str, Enum):
    """Represents the types of rating supported.

    This enum is used to track the possible rating definition:
        - EXPLICIT: The rating can be in a range, usually using floats.
            The scores will be read from raw transaction data.
        - IMPLICIT: The rating will all be set to 1.
            These scores won't be read from raw transaction data.
    """

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class SplittingStrategies(str, Enum):
    """Represents the types of splitting strategies supported.

    This enum is used to track the possible splitting strategies:
        - NONE: The splitting will not be performed.
        - RANDOM: The splitting will be random.
            A seed will bi used to ensure reproducibility.
        - LEAVE_ONE_OUT: The splitting will remove just one element.
            The elements chosen will be the same if a seed is set.
        - TEMPORAL: The splitting will be based on the timestamp.
            Timestamps will be mandatory if this strategy is chosen.
    """

    NONE = "none"
    RANDOM = "random"
    LEAVE_ONE_OUT = "leave-one-out"
    TEMPORAL = "temporal"


class SearchAlgorithms(str, Enum):
    """Represents the types of search algorithms supported.

    This enum is used to track the possible search algorithms:
        - GRID: The exhaustive search of all possible combinations hyperparameters.
        - RANDOM: The random search over all the search space.
        - HYPEROPT: The TPE algorithm implemented inside HyperOpt.
        - OPTUNA: The Optuna optimization algorithm.
        - BOHB: The Bandit-Wise and Bayesian optimization algorithm.
    """

    GRID = "grid"
    RANDOM = "random"
    HYPEROPT = "hopt"
    OPTUNA = "optuna"
    BOHB = "bohb"


class Schedulers(str, Enum):
    """Represents the types of schedulers supported.

    This enum is used to track the possible schedulers:
        - FIFO: Classic First-In First-Out implementation.
        - ASHA: Scheduler that supports parallelism and early stopping.
    """

    FIFO = "fifo"
    ASHA = "asha"
