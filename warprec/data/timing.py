import math
import time
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from warprec.utils.logger import logger

# The steps of the data preparation, by the column of the time report that
# gives each one's seconds.
STEP_COLUMNS: Dict[str, str] = {
    "reading": "Data Reading Time (s)",
    "filtering": "Filtering Time (s)",
    "splitting": "Splitting Time (s)",
}
BUILDING_COLUMN = "Dataset Building Time (s)"

# The structures the dataset building is made of, by the name a dataset records
# each one under and the column of the time report that gives its seconds.
STRUCTURE_COLUMNS: Dict[str, str] = {
    "dataset": "Dataset Core Building Time (s)",
    "interactions": "Interactions Building Time (s)",
    "sessions": "Sessions Building Time (s)",
    "side": "Side Information Building Time (s)",
    "context": "Context Building Time (s)",
    "knowledge": "Knowledge Graph Building Time (s)",
    "multimodal": "Multimodal Building Time (s)",
    "evaluation": "Evaluation Data Building Time (s)",
}


@contextmanager
def timed(store: Dict[str, float], key: str) -> Iterator[None]:
    """Add the seconds the enclosed block takes to a running total.

    Measured with a monotonic clock, so that the totals of several blocks, or of
    the same block run several times, can be added together.

    Args:
        store (Dict[str, float]): The totals, by key.
        key (str): The total the block adds to.

    Yields:
        None: The block runs while the clock does.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + time.perf_counter() - start


@dataclass
class DataPreparationTimes:
    """Where the preparation of a run's data spent its time.

    Every step and structure is measured where its work is done, so a
    preparation run in a Ray task measures only the task's own work: none of
    Ray's start-up or scheduling. A step or structure the run did not perform
    (no filtering configured, no knowledge graph, ...) has no time at all.

    Attributes:
        seconds (Dict[str, float]): The seconds of each step ('reading',
            'filtering', 'splitting') and structure (the keys of
            STRUCTURE_COLUMNS), summed over every dataset that built it.
    """

    seconds: Dict[str, float] = field(default_factory=dict)

    def measure(self, key: str) -> AbstractContextManager[None]:
        """Time a block as one of the steps or structures.

        Args:
            key (str): The step or structure the block belongs to.

        Returns:
            AbstractContextManager[None]: The context timing the block.
        """
        return timed(self.seconds, key)

    def add_build_times(self, build_times: Dict[str, float]) -> None:
        """Add the structures one dataset built to the totals.

        Args:
            build_times (Dict[str, float]): The seconds of each structure, as
                'Dataset.get_build_times' gives them.
        """
        for key, value in build_times.items():
            self.seconds[key] = self.seconds.get(key, 0.0) + value

    def building_seconds(self) -> Optional[float]:
        """The seconds spent building the datasets, every structure included.

        Returns:
            Optional[float]: The sum of the structures, or None if no dataset
                was built.
        """
        built = [self.seconds[key] for key in STRUCTURE_COLUMNS if key in self.seconds]
        return sum(built) if built else None

    def columns(self) -> Dict[str, float]:
        """The time report's columns, NaN for what the run did not do.

        Returns:
            Dict[str, float]: The seconds of each step, the dataset building as a
                whole and each of its structures, by column.
        """
        building = self.building_seconds()
        return {
            **{
                column: self.seconds.get(key, math.nan)
                for key, column in STEP_COLUMNS.items()
            },
            BUILDING_COLUMN: math.nan if building is None else building,
            **{
                column: self.seconds.get(key, math.nan)
                for key, column in STRUCTURE_COLUMNS.items()
            },
        }

    def log(self) -> None:
        """Log the seconds of every step and structure the run went through."""
        lines: List[str] = [
            f"{column}: {value:.3f}"
            for column, value in self.columns().items()
            if not math.isnan(value)
        ]
        if lines:
            logger.msg("Data preparation breakdown: " + ", ".join(lines))
