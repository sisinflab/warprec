from typing import Tuple, Optional, Union

from pandas import DataFrame
from warprec.utils.config import Configuration, SplittingConfig
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
        strategy: Optional[SplittingStrategies] = None,
        test_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_k: Optional[int] = None,
        val_k: Optional[int] = None,
        timestamp: Optional[Union[int, str]] = None,
        seed: int = 42,
    ) -> Tuple[DataFrame, DataFrame, Optional[DataFrame]]:
        """The main method of the class. This method must be called to split the data.

        When called, this method will return the splitting calculated by
        the splitting method selected in the configuration file.

        This method accepts transaction data, and will return a TransactionDataset object.

        A transaction is defined by at least a user_id, an item_id.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            strategy (Optional[SplittingStrategies]): The splitting strategy to use.
            test_ratio (Optional[float]): The test set size.
            val_ratio (Optional[float]): The validation set size.
            test_k (Optional[int]): The k value for test set.
            val_k (Optional[int]): The k value for validation set.
            timestamp (Optional[Union[int, str]]): The timestamp to be used for the test set.
                Either an integer or 'best'.
            seed (int): The seed used during splitting.

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
                strategy=strategy,
                test_ratio=test_ratio,
                val_ratio=val_ratio,
                test_k=test_k,
                val_k=val_k,
                timestamp=timestamp,
                seed=seed,
            )

        logger.msg(
            f"Starting splitting process with {split_config.strategy.value} splitting strategy."
        )

        # Get indexes using chosen strategy
        splitter = splitting_registry.get(split_config.strategy)
        idxs = splitter.split(
            data=data,
            test_ratio=split_config.test_ratio,
            val_ratio=split_config.val_ratio,
            test_k=split_config.test_k,
            val_k=split_config.val_k,
            timestamp=split_config.timestamp,
            seed=split_config.seed,
        )

        # Logging of splitting process
        logger.msg("Splitting process over.")

        # Define train/test/val subset taking into account
        # only user and items present in train set.
        train_set = data.iloc[idxs[0]]
        test_set = data.iloc[idxs[1]]
        test_set = test_set[
            test_set.iloc[:, 0].isin(
                train_set.iloc[:, 0]
            )  # Filter user from training set
            & test_set.iloc[:, 1].isin(
                train_set.iloc[:, 1]
            )  # Filter items from training set
        ]
        if idxs[2]:
            val_set = data.iloc[idxs[2]]
            val_set = val_set[
                val_set.iloc[:, 0].isin(
                    train_set.iloc[:, 0]
                )  # Filter user from training set
                & val_set.iloc[:, 1].isin(
                    train_set.iloc[:, 1]
                )  # Filter items from training set
            ]
            return (train_set, test_set, val_set)
        return (train_set, test_set, None)

    def split_context(
        self, data: DataFrame
    ) -> Tuple[DataFrame, DataFrame, Optional[DataFrame]]:
        """This function will be used to split context data."""
        raise NotImplementedError
