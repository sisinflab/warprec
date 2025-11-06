import time
from typing import Tuple, Optional, Union, List

from pandas import DataFrame
from warprec.utils.enums import SplittingStrategies
from warprec.utils.registry import splitting_registry
from warprec.utils.logger import logger


class Splitter:
    """Splitter class will handle the splitting of the data."""

    def split_transaction(
        self,
        data: DataFrame,
        user_id_label: str = "user_id",
        item_id_label: str = "item_id",
        rating_label: str = "rating",
        timestamp_label: str = "timestamp",
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
    ) -> Tuple[
        DataFrame, Optional[List[Tuple[DataFrame, DataFrame]] | DataFrame], DataFrame
    ]:
        """The main method of the class. This method must be called to split the data.

        When called, this method will return the splitting calculated by
        the splitting method selected in the configuration file.

        This method accepts transaction data, and will return the DataFrames of split data.

        A transaction is defined by at least a user_id, an item_id.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            user_id_label (str): The user_id label.
            item_id_label (str): The item_id label.
            rating_label (str): The rating label.
            timestamp_label (str): The timestamp label.
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
            Tuple[DataFrame, Optional[List[Tuple[DataFrame, DataFrame]] | DataFrame], DataFrame]:
                - DataFrame: The original train data, used to train
                    the final model of the experiment.
                - Optional[List[Tuple[DataFrame, DataFrame]] | DataFrame]: Either return a list of tuples
                    - DataFrame: The train data used to train the model.
                    - DataFrame: The validation data used to evaluate
                        the model during training.
                    or just a single DataFrame representing the validation set.
                - DataFrame: The unique test data, used at the end of
                    the experiment to evaluate the model.
        """
        # Test set
        split_process_start_time = time.time()
        logger.msg(
            f"Starting test splitting process with {test_strategy.value} splitting strategy."
        )
        test_split_time_start = time.time()
        original_train_set, test_set = self.process_split(
            data,
            test_strategy,
            user_id_label=user_id_label,
            item_id_label=item_id_label,
            rating_label=rating_label,
            timestamp_label=timestamp_label,
            ratio=test_ratio,
            k=test_k,
            folds=test_folds,
            timestamp=test_timestamp,
            seed=test_seed,
        )[0]
        test_split_time = time.time() - test_split_time_start
        logger.msg(f"Test splitting completed in : {test_split_time:.2f}s")

        # Optional validation folding
        validation_folds: List[Tuple[DataFrame, Optional[DataFrame]]] = []
        if val_strategy is not None:
            logger.msg(
                f"Starting validation splitting process with {val_strategy.value} splitting strategy."
            )
            validation_split_time_start = time.time()
            folds = self.process_split(
                original_train_set,
                val_strategy,
                user_id_label=user_id_label,
                item_id_label=item_id_label,
                rating_label=rating_label,
                timestamp_label=timestamp_label,
                ratio=val_ratio,
                k=val_k,
                folds=val_folds,
                timestamp=val_timestamp,
                seed=val_seed,
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
        self.filter_sets(original_train_set, test_set)

        if len(validation_folds) == 0:
            # CASE 1: Only train and test set
            return (original_train_set, None, test_set)

        if len(validation_folds) == 1:
            # CASE 2: Train/Validation/Test
            train_set, validation_set = validation_folds[0]
            self.filter_sets(train_set, test_set)
            return (train_set, validation_set, test_set)

        # Filter out each validation set based on
        # corresponding train set
        for train, validation in validation_folds:
            self.filter_sets(train, validation)

        # CASE 3: N folds of train and validation + the test set
        return (original_train_set, validation_folds, test_set)

    def process_split(
        self,
        data: DataFrame,
        strategy: SplittingStrategies,
        user_id_label: str = "user_id",
        item_id_label: str = "item_id",
        rating_label: str = "rating",
        timestamp_label: str = "timestamp",
        ratio: Optional[float] = None,
        k: Optional[int] = None,
        folds: Optional[int] = None,
        timestamp: Optional[Union[int, str]] = None,
        seed: int = 42,
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """Process the splitting based on the selected strategy.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            strategy (SplittingStrategies): The splitting strategy to use.
            user_id_label (str): The user_id label.
            item_id_label (str): The item_id label.
            rating_label (str): The rating label.
            timestamp_label (str): The timestamp label.
            ratio (Optional[float]): The ratio value.
            k (Optional[int]): The k value.
            folds (Optional[int]): The folds value.
            timestamp (Optional[Union[int, str]]): The timestamp to be used for the splitting.
                Either an integer or 'best'.
            seed (int): The seed value. Defaults to 42.

        Returns:
            List[Tuple[DataFrame, DataFrame]]: A list of tuples containing the train and evaluation sets.
        """
        splitting_strategy = splitting_registry.get(strategy.value)
        split = splitting_strategy(
            data,
            user_id_label=user_id_label,
            item_id_label=item_id_label,
            rating_label=rating_label,
            timestamp_label=timestamp_label,
            ratio=ratio,
            k=k,
            folds=folds,
            timestamp=timestamp,
            seed=seed,
        )
        return split

    def filter_sets(self, train_set: DataFrame, evaluation_set: DataFrame):
        """Filter the evaluation set based on the train set.

        Args:
            train_set (DataFrame): The training set.
            evaluation_set (DataFrame): The evaluation set to be filtered.
        """
        mask = evaluation_set.iloc[:, 0].isin(
            train_set.iloc[:, 0]
        ) & evaluation_set.iloc[:, 1].isin(train_set.iloc[:, 1])
        evaluation_set.drop(index=evaluation_set.index[~mask], inplace=True)
