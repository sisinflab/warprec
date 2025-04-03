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


class ReadingMethods(str, Enum):
    """Represents the types of reding strategies supported.

    This enum is used to track the possible reading methods:
        - LOCAL: Read data locally.
    """

    LOCAL = "local"


class WritingMethods(str, Enum):
    """Represents the types of writing strategies supported.

    This enum is used to track the possible writing methods:
        - LOCAL: Writes results locally.
    """

    LOCAL = "local"


class Similarities(str, Enum):
    """Represents the types of similarities supported.

    This enum is used to track the possible similarities:
        - COSINE: Cosine similarity.
        - DOT: Dot similarity.
        - EUCLIDEAN: Euclidean similarity.
        - MANHATTAN: Manhattan similarity.
    """

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class SearchAlgorithms(str, Enum):
    """Represents the types of search algorithms supported.

    This enum is used to track the possible search algorithms:
        - GRID: Performs grid search over all the parameters provided.
        - RANDOM: Random search over the param space.
        - HYPEROPT: Bayesian optimization using HyperOptOptimization.
        - OPTUNA: Optuna optimization, more information can
            be found at https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html
        - BOHB: BOHB optimization, more information can
            be found at https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.bohb.TuneBOHB.html
    """

    GRID = "grid"
    RANDOM = "random"
    HYPEROPT = "hopt"
    OPTUNA = "optuna"
    BOHB = "bohb"


class Schedulers(str, Enum):
    """Represents the types of schedulers supported.

    This enum is used to track the possible schedulers:
        - FIFO: Classic First In First Out trial optimization.
        - ASHA: ASHA Scheduler, more information can be found
            at https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.ASHAScheduler.html.
    """

    FIFO = "fifo"
    ASHA = "asha"


class SearchSpace(str, Enum):
    """Represents the types of search spaces supported.

    This enum is used to track the possible search spaces:
        - UNIFORM: Uniform search space.
        - QUNIFORM: Rounded Uniform search space.
        - LOGUNIFORM: Logarithmic Uniform search space.
        - QLOGUNIFORM: Rounded Logarithmic Uniform search space.
        - RANDN: Random search space.
        - QRANDN: Rounded Random search space.
        - RANDINT: Random Integer search space.
        - QRANDINT: Rounded Random Integer search space.
        - LOGRANDINT: Logarithmic Random Integer search space.
        - QLOGRANDINT: Rounded Logarithmic Random Integer search space.
        - CHOICE: Discrete search space.
        - GRID: Exhaustive discrete search space for grid search.
    """

    UNIFORM = "uniform"
    QUNIFORM = "quniform"
    LOGUNIFORM = "loguniform"
    QLOGUNIFORM = "qloguniform"
    RANDN = "randn"
    QRANDN = "qrandn"
    RANDINT = "randint"
    QRANDINT = "qrandint"
    LOGRANDINT = "lograndint"
    QLOGRANDINT = "qlograndint"
    CHOICE = "choice"
    GRID = "grid"
