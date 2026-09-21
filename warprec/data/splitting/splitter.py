import time
from typing import Tuple, Optional, List, Any

import narwhals as nw
from narwhals.typing import FrameT
from narwhals.dataframe import DataFrame

from warprec.data.schema import ColumnLabels, SplitSpec
from warprec.utils.registry import splitting_registry
from warprec.utils.logger import logger


SplitResult = Tuple[
    DataFrame[Any],
    Optional[List[Tuple[DataFrame[Any], DataFrame[Any]]] | DataFrame[Any]],
    DataFrame[Any],
]


class Splitter:
    """Splitter class will handle the splitting of the data."""

    def split_transaction(
        self,
        data: FrameT,
        labels: Optional[ColumnLabels] = None,
        test: Optional[SplitSpec] = None,
        validation: Optional[SplitSpec] = None,
    ) -> SplitResult:
        """The main method of the class. This method must be called to split the data.

        When called, this method will return the splitting calculated by
        the splitting method selected in the configuration file.

        This method accepts transaction data, and will return the DataFrames of split data.

        A transaction is defined by at least a user_id, an item_id.

        Args:
            data (FrameT): The DataFrame to be splitted.
            labels (Optional[ColumnLabels]): The names the four core columns carry.
                Defaults to WarpRec's own schema.
            test (Optional[SplitSpec]): The criterion the test split follows.
            validation (Optional[SplitSpec]): The criterion the validation split
                follows. Left out, no validation set is produced.

        Returns:
            SplitResult:
                The train set, the validation set (a single frame, a list of
                folds, or None) and the test set.
        """
        data = nw.from_native(data, pass_through=True)
        labels = labels or ColumnLabels()
        test = test or SplitSpec()

        # Test set
        split_process_start_time = time.time()
        test_strategy = test.resolved_strategy()
        logger.msg(
            f"Starting test splitting process with {test_strategy.value} splitting strategy."
        )
        test_split_time_start = time.time()
        original_train_set, test_set = self.process_split(data, test, labels)[0]
        test_split_time = time.time() - test_split_time_start
        logger.msg(f"Test splitting completed in : {test_split_time:.2f}s")

        # Optional validation folding
        validation_folds: List[Tuple[DataFrame[Any], Optional[DataFrame[Any]]]] = []
        if validation is not None and validation.strategy is not None:
            val_strategy = validation.resolved_strategy()
            logger.msg(
                f"Starting validation splitting process with {val_strategy.value} splitting strategy."
            )
            validation_split_time_start = time.time()
            folds = self.process_split(original_train_set, validation, labels)
            for train, val_set in folds:
                validation_folds.append((train, val_set))
            validation_split_time = time.time() - validation_split_time_start
            logger.msg(
                f"Validation splitting completed in : {validation_split_time:.2f}s"
            )

        # Logging of splitting process
        split_process_time = time.time() - split_process_start_time
        logger.positive(f"Splitting process over in {split_process_time:.2f}s.")

        # Filter out the test set
        test_set = self.filter_sets(
            original_train_set, test_set, labels.user_id, labels.item_id, "Test"
        )

        if len(validation_folds) == 0:
            # CASE 1: Only train and test set
            return (original_train_set, None, test_set)

        if len(validation_folds) == 1:
            # CASE 2: Train/Validation/Test
            train_set, validation_set = validation_folds[0]
            test_set = self.filter_sets(
                train_set, test_set, labels.user_id, labels.item_id, "Validation"
            )
            return (train_set, validation_set, test_set)

        # Filter out each validation set based on
        # corresponding train set
        for train, val_set in validation_folds:
            val_set = self.filter_sets(
                train, val_set, labels.user_id, labels.item_id, "Validation"
            )

        # CASE 3: N folds of train and validation + the test set
        return (original_train_set, validation_folds, test_set)

    def process_split(
        self,
        data: FrameT,
        spec: SplitSpec,
        labels: Optional[ColumnLabels] = None,
    ) -> List[Tuple[DataFrame[Any], DataFrame[Any]]]:
        """Process the splitting based on the selected strategy.

        Args:
            data (FrameT): The DataFrame to be splitted.
            spec (SplitSpec): The strategy and the parameters it reads.
            labels (Optional[ColumnLabels]): The names the four core columns carry.

        Returns:
            List[Tuple[DataFrame[Any], DataFrame[Any]]]: A list of tuples containing the train and evaluation sets.
        """
        labels = labels or ColumnLabels()
        splitting_strategy = splitting_registry.get(spec.resolved_strategy().value)
        split = splitting_strategy(
            data,
            user_id_label=labels.user_id,
            item_id_label=labels.item_id,
            rating_label=labels.rating,
            timestamp_label=labels.timestamp,
            ratio=spec.ratio,
            k=spec.k,
            folds=spec.folds,
            timestamp=spec.timestamp,
            seed=spec.seed,
        )
        return split

    def filter_sets(
        self,
        train_set: DataFrame[Any],
        evaluation_set: DataFrame[Any],
        user_id_label: str = "user_id",
        item_id_label: str = "item_id",
        eval_set_name: Optional[str] = None,
    ) -> DataFrame[Any]:
        """Filter the evaluation set based on the train set.

        Args:
            train_set (DataFrame[Any]): The training set.
            evaluation_set (DataFrame[Any]): The evaluation set to be filtered.
            user_id_label (str): The user ID label.
            item_id_label (str): The item ID label.
            eval_set_name (Optional[str]): The name of the evaluation set.
                Used for logging purposes.

        Returns:
            DataFrame[Any]: The filtered evaluation set.
        """
        train_users = train_set.select(user_id_label).unique()
        train_items = train_set.select(item_id_label).unique()

        # Save the evaluation transaction before filtering
        eval_transaction_count = len(evaluation_set)

        filtered_by_users = evaluation_set.join(
            train_users, on=user_id_label, how="inner"
        )

        filtered_final = filtered_by_users.join(
            train_items, on=item_id_label, how="inner"
        )

        # Log any filtering that happened
        if len(filtered_final) < eval_transaction_count:
            eval_set_name = (
                eval_set_name.capitalize() if eval_set_name is not None else "Eval"
            )
            logger.attention(
                f"{eval_set_name} set was not aligned with the training set. "
                f"Filtered out {eval_transaction_count - len(filtered_final)} transactions."
            )

        return filtered_final
