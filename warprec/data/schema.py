from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from scipy.sparse import csr_matrix

from warprec.utils.enums import RatingType, SplittingStrategies


@dataclass(frozen=True)
class ColumnLabels:
    """The names the four core columns carry in the data being processed.

    They travel together through every stage that has to address a column by
    name, which is why they are one value rather than four parallel arguments.

    Attributes:
        user_id (str): The name of the user id column.
        item_id (str): The name of the item id column.
        rating (str): The name of the rating column.
        timestamp (str): The name of the timestamp column.
    """

    user_id: str = "user_id"
    item_id: str = "item_id"
    rating: str = "rating"
    timestamp: str = "timestamp"


@dataclass(frozen=True)
class SplitSpec:
    """One split criterion: the strategy and the parameters it reads.

    A run describes two of these, one for the test split and one for the
    validation split, and they take the same shape. Keeping them as a single
    value is what stops the same six parameters appearing twice, once under a
    'test_' prefix and once under a 'val_' one.

    Attributes:
        strategy (Optional[Union[SplittingStrategies, str]]): The splitting strategy.
        ratio (Optional[float]): The ratio the strategy reads, when it takes one.
        k (Optional[int]): The k the strategy reads, when it takes one.
        folds (Optional[int]): The number of folds, for cross validation.
        timestamp (Optional[Union[int, str]]): The pivot timestamp, or 'best'.
        seed (int): The seed that makes a random strategy reproducible.
    """

    strategy: Optional[Union[SplittingStrategies, str]] = None
    ratio: Optional[float] = None
    k: Optional[int] = None
    folds: Optional[int] = None
    timestamp: Optional[Union[int, str]] = None
    seed: int = 42

    def resolved_strategy(self) -> Optional[SplittingStrategies]:
        """The strategy as an enum member, however it was written.

        Returns:
            Optional[SplittingStrategies]: The strategy, or None when unset.
        """
        if isinstance(self.strategy, str):
            return SplittingStrategies(self.strategy)
        return self.strategy


@dataclass(frozen=True)
class ContextSpec:
    """Everything the data layer needs to know about the contextual columns.

    The four pieces are decided together when the dataset is read and are then
    needed together by every structure that carries contexts, so they travel as
    one value rather than as four parallel arguments.

    Attributes:
        labels (Tuple[str, ...]): The contextual columns, in order.
        types (Dict[str, str]): The declared type of each column.
        field_types (Dict[str, str]): How each field is encoded, 'token',
            'float' or 'seq'.
        separators (Dict[str, str]): The separator of each multi-valued column.
        max_len (int): The widest multi-valued field, or 1 when there is none.
    """

    labels: Tuple[str, ...] = ()
    types: Dict[str, str] = field(default_factory=dict)
    field_types: Dict[str, str] = field(default_factory=dict)
    separators: Dict[str, str] = field(default_factory=dict)
    max_len: int = 1

    def label_list(self) -> List[str]:
        """The contextual columns as the list the entities store.

        Returns:
            List[str]: The labels.
        """
        return list(self.labels)


@dataclass(frozen=True)
class SignalOptions:
    """The options that decide what a model is trained on, not what is read.

    Attributes:
        rating_type (RatingType): Whether the feedback is implicit or explicit.
        duplicates (str): How repeated (user, item) rows are aggregated.
        negative_sampling (str): How negatives are drawn during training.
        sequence_pooling (str): How a multi-valued field's values are pooled.
        batch_size (int): The batch size the structures are built for.
    """

    rating_type: RatingType = RatingType.IMPLICIT
    duplicates: str = "max"
    negative_sampling: str = "uniform"
    sequence_pooling: str = "mean"
    batch_size: int = 1024


@dataclass(frozen=True)
class SideData:
    """The item attributes and the cluster assignments, when a run has them.

    Attributes:
        frame (Optional[Any]): The raw side information frame.
        matrix (Optional[csr_matrix]): The {item x feature} content matrix.
        user_cluster (Optional[dict]): The user cluster assignments.
        item_cluster (Optional[dict]): The item cluster assignments.
    """

    frame: Optional[Any] = None
    matrix: Optional[csr_matrix] = None
    user_cluster: Optional[dict] = None
    item_cluster: Optional[dict] = None
