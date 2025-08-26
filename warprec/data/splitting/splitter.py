import time
from typing import Tuple, Optional, Union, List

from pandas import DataFrame
from warprec.utils.config import (
    TrainConfiguration,
    DesignConfiguration,
    SplittingConfig,
    SplitStrategy,
)
from warprec.utils.enums import SplittingStrategies
from warprec.utils.registry import splitting_registry
from warprec.utils.logger import logger


class Splitter:
    """Splitter class will handle the splitting of the data.

    Args:
        config (TrainConfiguration | DesignConfiguration): The configuration file.

    Attributes:
        config (TrainConfiguration | DesignConfiguration): The configuration file.
    """

    config: TrainConfiguration | DesignConfiguration = None

    def __init__(self, config: TrainConfiguration | DesignConfiguration = None):
        if config:
            self.config = config

    def split_transaction(
        self,
        data: DataFrame,
        test_strategy: Optional[SplittingStrategies] = None,
        test_ratio: Optional[float] = None,
        test_k: Optional[int] = None,
        test_folds: Optional[int] = None,
        test_timestamp: Optional[Union[int, str]] = None,
        test_seed: int = 42,
        val_strategy: Optional[SplittingStrategies] = None,
        val_ratio: Optional[float] = None,
        val_k: Optional[int] = None,
        val_folds: Optional[int] = None,
        val_timestamp: Optional[Union[int, str]] = None,
        val_seed: int = 42,
    ) -> Tuple[DataFrame, Optional[List[Tuple[DataFrame, DataFrame]]], DataFrame]:
        """The main method of the class. This method must be called to split the data.

        When called, this method will return the splitting calculated by
        the splitting method selected in the configuration file.

        This method accepts transaction data, and will return the DataFrames of split data.

        A transaction is defined by at least a user_id, an item_id.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            test_strategy (Optional[SplittingStrategies]): The splitting strategy to use for test set.
            test_ratio (Optional[float]): The ratio value for test set.
            test_k (Optional[int]): The k value for test set.
            test_folds (Optional[int]): The folds value for test set.
            test_timestamp (Optional[Union[int, str]]): The timestamp to be used for the test set.
                Either an integer or 'best'.
            test_seed (int): The seed value for test set. Defaults to 42.
            val_strategy (Optional[SplittingStrategies]): The splitting strategy to use for validation set.
            val_ratio (Optional[float]): The ratio value for validation set.
            val_k (Optional[int]): The k value for validation set.
            val_folds (Optional[int]): The folds value for validation set.
            val_timestamp (Optional[Union[int, str]]): The timestamp to be used for the validation set.
                Either an integer or 'best'.
            val_seed (int): The seed value for validation set.  Defaults to 42.

        Returns:
            Tuple[DataFrame, Optional[List[Tuple[DataFrame, DataFrame]]], DataFrame]:
                - DataFrame: The original train data, used to train
                    the final model of the experiment.
                - Optional[List[Tuple[DataFrame, DataFrame]]]:
                    - DataFrame: The train data used to train the model.
                    - DataFrame: The validation data used to evaluate
                        the model during training.
                - DataFrame: The unique test data, used at the end of
                    the experiment to evaluate the model.
        """
        if self.config:
            split_config = self.config.splitter
        else:
            split_config = SplittingConfig(
                test_splitting=SplitStrategy(
                    strategy=test_strategy,
                    ratio=test_ratio,
                    k=test_k,
                    folds=test_folds,
                    timestamp=test_timestamp,
                    seed=test_seed,
                ),
                validation_splitting=SplitStrategy(
                    strategy=val_strategy,
                    ratio=val_ratio,
                    k=val_k,
                    folds=val_folds,
                    timestamp=val_timestamp,
                    seed=val_seed,
                ),
            )
        # Test set
        split_process_start_time = time.time()
        logger.msg(
            f"Starting test splitting process with {split_config.test_splitting.strategy.value} splitting strategy."
        )
        test_split_time_start = time.time()
        original_train_set, test_set = self._process_split(
            data, split_config.test_splitting
        )[0]
        test_split_time = time.time() - test_split_time_start
        logger.msg(f"Test splitting completed in : {test_split_time:.2f}s")

        # Optional validation folding
        validation_folds: List[Tuple[DataFrame, Optional[DataFrame]]] = []
        if split_config.validation_splitting.strategy is not None:
            logger.msg(
                f"Starting validation splitting process with {split_config.validation_splitting.strategy.value} splitting strategy."
            )
            validation_split_time_start = time.time()
            folds = self._process_split(
                original_train_set, split_config.validation_splitting
            )
            for train, validation in folds:
                validation_folds.append((train, validation))
            validation_split_time = time.time() - validation_split_time_start
            logger.msg(
                f"Validation splitting completed in : {validation_split_time:.2f}s"
            )

        # Logging of splitting process
        split_process_time = time.time() - split_process_start_time
        logger.positive(f"Splitting process over in {split_process_time:.2f}s.")

        # Filter out the test set
        self._filter_sets(original_train_set, test_set)

        if len(validation_folds) == 0:
            # CASE 1: Only train and test set
            return (original_train_set, None, test_set)

        # Filter out each validation set based on
        # corresponding train set
        for train, validation in validation_folds:
            self._filter_sets(train, validation)

        # CASE 2: N folds of train and validation + the test set
        return (original_train_set, validation_folds, test_set)

    def split_context(
        self, data: DataFrame
    ) -> Tuple[DataFrame, DataFrame, Optional[DataFrame]]:
        """This function will be used to split context data."""
        raise NotImplementedError

    def _process_split(
        self, data: DataFrame, split_strategy: SplitStrategy
    ) -> List[Tuple[DataFrame, DataFrame]]:
        strategy = splitting_registry.get(split_strategy.strategy)
        folds = strategy(data, **split_strategy.model_dump())
        return folds

    def _filter_sets(self, train_set: DataFrame, evaluation_set: DataFrame):
        mask = evaluation_set.iloc[:, 0].isin(
            train_set.iloc[:, 0]
        ) & evaluation_set.iloc[:, 1].isin(train_set.iloc[:, 1])
        evaluation_set.drop(index=evaluation_set.index[~mask], inplace=True)
