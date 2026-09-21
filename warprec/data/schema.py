from dataclasses import dataclass
from typing import Optional, Union

from warprec.utils.enums import SplittingStrategies


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
