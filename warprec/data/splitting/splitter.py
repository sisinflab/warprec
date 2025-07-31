import time
from typing import Tuple, Optional, Union

from pandas import DataFrame
from warprec.utils.config import Configuration, SplittingConfig, SplitStrategy
from warprec.utils.enums import SplittingStrategies
from warprec.utils.registry import splitting_registry
from warprec.utils.logger import logger


class Splitter:
    """Splitter class will handle the splitting of the data.

    Args:
        config (Configuration): The configuration file.

    Attributes:
        config (Configuration): The configuration file.
    """

    config: Configuration = None

    def __init__(self, config: Configuration = None):
        if config:
            self.config = config

    def split_transaction(
        self,
        data: DataFrame,
        test_strategy: Optional[SplittingStrategies] = None,
        test_ratio: Optional[float] = None,
        test_k: Optional[int] = None,
        test_timestamp: Optional[Union[int, str]] = None,
        test_seed: int = 42,
        val_strategy: Optional[SplittingStrategies] = None,
        val_ratio: Optional[float] = None,
        val_k: Optional[int] = None,
        val_timestamp: Optional[Union[int, str]] = None,
        val_seed: int = 42,
    ) -> Tuple[DataFrame, DataFrame, Optional[DataFrame]]:
        """The main method of the class. This method must be called to split the data.

        When called, this method will return the splitting calculated by
        the splitting method selected in the configuration file.

        This method accepts transaction data, and will return a TransactionDataset object.

        A transaction is defined by at least a user_id, an item_id.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_strategy (Optional[SplittingStrategies]): The splitting strategy to use for test set.
            test_ratio (Optional[float]): The ratio value for test set.
            test_k (Optional[int]): The k value for test set.
            test_timestamp (Optional[Union[int, str]]): The timestamp to be used for the test set.
                Either an integer or 'best'.
            test_seed (int): The seed value for test set. Defaults to 42.
            val_strategy (Optional[SplittingStrategies]): The splitting strategy to use for validation set.
            val_ratio (Optional[float]): The ratio value for validation set.
            val_k (Optional[int]): The k value for validation set.
            val_timestamp (Optional[Union[int, str]]): The timestamp to be used for the validation set.
                Either an integer or 'best'.
            val_seed (int): The seed value for validation set.  Defaults to 42.

        Returns:
            Tuple[DataFrame, DataFrame, Optional[DataFrame]]:
                DataFrame: The train data.
                DataFrame: The test data.
                Optional[DataFrame]: The validation data if needed.
        """
        if self.config:
            split_config = self.config.splitter
        else:
            split_config = SplittingConfig(
                test_splitting=SplitStrategy(
                    strategy=test_strategy,
                    ratio=test_ratio,
                    k=test_k,
                    timestamp=test_timestamp,
                    seed=test_seed,
                ),
                validation_splitting=SplitStrategy(
                    strategy=val_strategy,
                    ratio=val_ratio,
                    k=val_k,
                    timestamp=val_timestamp,
                    seed=val_seed,
                ),
            )
        data_to_split = data
        train_set: DataFrame = None
        validation_set: DataFrame = None
        test_set: DataFrame = None

        if split_config.test_splitting.strategy is not None:
            train_set, test_set = self._process_split(
                data_to_split, split_config.test_splitting, "Test"
            )
            data_to_split = train_set
        if split_config.validation_splitting.strategy is not None:
            train_set, validation_set = self._process_split(
                data_to_split, split_config.validation_splitting, "Validation"
            )

        # Logging of splitting process
        logger.positive("Splitting process over.")

        if test_set is not None:
            test_set = test_set[
                test_set.iloc[:, 0].isin(
                    train_set.iloc[:, 0]
                )  # Filter user from training set
                & test_set.iloc[:, 1].isin(
                    train_set.iloc[:, 1]
                )  # Filter items from training set
            ]

        if validation_set is not None:
            validation_set = validation_set[
                validation_set.iloc[:, 0].isin(
                    train_set.iloc[:, 0]
                )  # Filter user from training set
                & validation_set.iloc[:, 1].isin(
                    train_set.iloc[:, 1]
                )  # Filter items from training set
            ]

        return (train_set, test_set, validation_set)

    def split_context(
        self, data: DataFrame
    ) -> Tuple[DataFrame, DataFrame, Optional[DataFrame]]:
        """This function will be used to split context data."""
        raise NotImplementedError

    def _process_split(
        self, data: DataFrame, split_strategy: SplitStrategy, set_type: str
    ):
        logger.msg(
            f"Starting {set_type} splitting process with {split_strategy.strategy.value} splitting strategy."
        )
        strategy = splitting_registry.get(split_strategy.strategy)
        test_split_time_start = time.time()
        first_partition, second_partition = strategy(
            data, **split_strategy.model_dump()
        )
        test_split_time = time.time() - test_split_time_start

        logger.msg(f"{set_type} splitting completed in : {test_split_time:.2f}s")
        return first_partition, second_partition
