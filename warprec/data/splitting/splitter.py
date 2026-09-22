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
        cold = self.cold_dimension(test)

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
            original_train_set,
            test_set,
            labels.user_id,
            labels.item_id,
            "Test",
            cold_dimension=cold,
        )

        if len(validation_folds) == 0:
            # CASE 1: Only train and test set
            return (original_train_set, None, test_set)

        if len(validation_folds) == 1:
            # CASE 2: Train/Validation/Test
            train_set, validation_set = validation_folds[0]
            test_set = self.filter_sets(
                train_set,
                test_set,
                labels.user_id,
                labels.item_id,
                "Validation",
                cold_dimension=cold,
            )
            return (train_set, validation_set, test_set)

        # Filter out each validation set based on
        # corresponding train set
        for train, val_set in validation_folds:
            val_set = self.filter_sets(
                train,
                val_set,
                labels.user_id,
                labels.item_id,
                "Validation",
                cold_dimension=cold,
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

    @staticmethod
    def cold_dimension(spec: SplitSpec) -> Optional[str]:
        """Which side of the catalogue a split holds out entirely, if any.

        Args:
            spec (SplitSpec): The criterion the split follows.

        Returns:
            Optional[str]: 'item', 'user' or None.
        """
        strategy = spec.resolved_strategy()
        if strategy is None:
            return None
        return getattr(
            splitting_registry.get_class(strategy.value), "COLD_DIMENSION", None
        )

    def filter_sets(
        self,
        train_set: DataFrame[Any],
        evaluation_set: DataFrame[Any],
        user_id_label: str = "user_id",
        item_id_label: str = "item_id",
        eval_set_name: Optional[str] = None,
        cold_dimension: Optional[str] = None,
    ) -> DataFrame[Any]:
        """Filter the evaluation set based on the train set.

        An entity the model never saw cannot be scored, so it is normally dropped
        from the evaluation set. A cold-start protocol inverts that: the whole
        point is to ask about entities training never saw, and filtering them out
        would empty the evaluation set entirely. ``cold_dimension`` names the side
        that is unseen on purpose and must therefore survive.

        Args:
            train_set (DataFrame[Any]): The training set.
            evaluation_set (DataFrame[Any]): The evaluation set to be filtered.
            user_id_label (str): The user ID label.
            item_id_label (str): The item ID label.
            eval_set_name (Optional[str]): The name of the evaluation set.
                Used for logging purposes.
            cold_dimension (Optional[str]): The side held out on purpose, 'item',
                'user' or None. That side is not filtered.

        Returns:
            DataFrame[Any]: The filtered evaluation set.
        """
        # Save the evaluation transaction before filtering
        eval_transaction_count = len(evaluation_set)

        filtered_final = evaluation_set
        if cold_dimension != "user":
            train_users = train_set.select(user_id_label).unique()
            filtered_final = filtered_final.join(
                train_users, on=user_id_label, how="inner"
            )

        if cold_dimension != "item":
            train_items = train_set.select(item_id_label).unique()
            filtered_final = filtered_final.join(
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
