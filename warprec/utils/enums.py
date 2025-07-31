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
        - TEMPORAL_HOLDOUT: The splitting will be based on the timestamp and ratio.
            Timestamps will be mandatory if this strategy is chosen.
        - TEMPORAL_LEAVE_K_OUT: The splitting will be based on the timestamp and k.
            Timestamps will be mandatory if this strategy is chosen.
        - TIMESTAMP_SLICING: The splitting will be based on a timestamp. Either fixed or 'best'.
            The timestamp will be mandatory if this strategy is chosen.
        - RANDOM_HOLDOUT: The splitting will be based on a ratio. If a seed has been provided,
            the splitting will be deterministic.
        - RANDOM_LEAVE_K_OUT: The splitting will be based on k. If a seed has been provided,
            the splitting will be deterministic.
    """

    TEMPORAL_HOLDOUT = "temporal_holdout"
    TEMPORAL_LEAVE_K_OUT = "temporal_leave_k_out"
    TIMESTAMP_SLICING = "timestamp_slicing"
    RANDOM_HOLDOUT = "random_holdout"
    RANDOM_LEAVE_K_OUT = "random_leave_k_out"


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
        - HAVERSINE: Haversine similarity.
    """

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    HAVERSINE = "haversine"


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


class MetricBlock(str, Enum):
    """Represents the types metric blocks that can be precomputed. A metric block is
    a series of operation shared between metrics that can be computed one time and used by more
    than one metric for efficiency.

    This enum is used to track the possible metric blocks:
        - BINARY_RELEVANCE: The relevance represented as a binary tensor [0, 1].
        - DISCOUNTED_RELEVANCE: The relevance represented as a discounted tensor.
        - VALID_USERS: The number of users with at least one relevant item.
        - TOP_K_INDICES: The indices of the top k predictions of the model.
        - TOP_K_VALUES: The values of the top k predictions of the model.
        - TOP_K_BINARY_RELEVANCE: The relevance of the top k predictions as a binary tensor [0, 1].
        - TIO_K_DISCOUNTED_RELEVANCE: The relevance of the top k predictions as a discounted tensor.
    """

    BINARY_RELEVANCE = "binary_relevance"
    DISCOUNTED_RELEVANCE = "discounted_relevance"
    VALID_USERS = "valid_users"
    TOP_K_INDICES = "top_k_indices"
    TOP_K_VALUES = "top_k_values"
    TOP_K_BINARY_RELEVANCE = "top_k_binary_relevance"
    TOP_K_DISCOUNTED_RELEVANCE = "top_k_discounted_relevance"
