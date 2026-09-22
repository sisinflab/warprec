# pylint: disable=too-few-public-methods
# mypy: disable-error-code=override
from typing import Tuple, Any, Optional, Union, List
from abc import ABC

import numpy as np
import narwhals as nw
from narwhals.typing import FrameT
from narwhals.dataframe import DataFrame

from warprec.utils.enums import SplittingStrategies
from warprec.utils.registry import splitting_registry


class SplittingStrategy(ABC):
    """Abstract definition of a splitting strategy.

    Attributes:
        COLD_DIMENSION (Optional[str]): Which side of the catalogue the strategy
            holds out entirely, 'item', 'user' or None. A strategy that names one
            produces an evaluation set whose entities are absent from training on
            purpose, which the rest of the pipeline has to be told not to discard.
    """

    COLD_DIMENSION: Optional[str] = None

    def _prepare_stable_data(self, data: FrameT) -> Tuple[DataFrame[Any], str]:
        """Helper method to ensure data has a stable row index for tie-breaking.
        Returns the modified data and the name of the index column.
        """
        # Make the DataFrame eager in case of LazyFrames
        if isinstance(data, nw.LazyFrame):
            materialized_data = data.collect()  # type: ignore[assignment]
        else:
            materialized_data = data

        index_col = "__original_row_index__"
        return materialized_data.with_row_index(name=index_col), index_col

    def __call__(
        self, data: FrameT, **kwargs: Any
    ) -> List[Tuple[DataFrame[Any], DataFrame[Any]]]:
        """This method will split the data in train/eval split.

        Args:
            data (FrameT): The FrameT to be splitted.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame[Any], DataFrame[Any]]]:
                - DataFrame[Any]: First partition of splitted data.
                - DataFrame[Any]: Second partition of splitted data.
        """


@splitting_registry.register(SplittingStrategies.TEMPORAL_HOLDOUT)
class TemporalHoldoutSplit(SplittingStrategy):
    """The definition of a temporal holdout splitting strategy.

    In case the timestamp will not be provided, the former order
    of the transactions will be used.
    """

    def __call__(
        self,
        data: FrameT,
        user_id_label: str = "user_id",
        timestamp_label: str = "timestamp",
        ratio: float = 0.2,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame[Any], DataFrame[Any]]]:
        """Method to split data in two partitions, using a timestamp or the
        original order.

        This method will split data based on time, using as test
        samples the more recent transactions.

        Args:
            data (FrameT): The original data in FrameT format.
            user_id_label (str): The user_id label.
            timestamp_label (str): The timestamp label.
            ratio (float): Percentage of data that will end up in the second partition.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame[Any], DataFrame[Any]]]:
                - DataFrame[Any]: First partition of splitted data.
                - DataFrame[Any]: Second partition of splitted data.
        """
        data_prep, idx_col = self._prepare_stable_data(data)

        # Deterministic sorting
        if timestamp_label in data_prep.columns:
            data_prep = data_prep.sort(by=[user_id_label, timestamp_label, idx_col])
            rank_target_col = timestamp_label
        else:
            data_prep = data_prep.sort(by=[user_id_label, idx_col])
            rank_target_col = idx_col

        # Compute the cumulative count
        data_with_stats = data_prep.with_columns(
            n_items=nw.len().over(user_id_label),
            rank=nw.col(rank_target_col).rank(method="ordinal").over(user_id_label),
        )

        # Masking
        cutoff = (nw.col("n_items") * (1 - ratio)).floor().cast(nw.Int64)
        train_mask = (nw.col("rank") <= cutoff) | (nw.col("rank") == 1)

        # Clean the output
        cols_to_drop = ["n_items", "rank", idx_col]
        train = data_with_stats.filter(train_mask).drop(cols_to_drop)
        test = data_with_stats.filter(~train_mask).drop(cols_to_drop)

        return [(train, test)]


@splitting_registry.register(SplittingStrategies.TEMPORAL_LEAVE_K_OUT)
class TemporalLeaveKOutSplit(SplittingStrategy):
    """The definition of a temporal leave k out splitting strategy.

    In case the timestamp will not be provided, the former order
    of the transactions will be used.
    """

    def __call__(
        self,
        data: FrameT,
        user_id_label: str = "user_id",
        timestamp_label: str = "timestamp",
        k: int = 1,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame[Any], DataFrame[Any]]]:
        """Method to split data in two partitions, using a timestamp or the
        original order.

        This method will split data based on time, using as test
        samples the more recent transactions.

        Args:
            data (FrameT): The original data in FrameT format.
            user_id_label (str): The user_id label.
            timestamp_label (str): The timestamp label.
            k (int): Number of transaction that will end up in the second partition.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame[Any], DataFrame[Any]]]:
                - DataFrame[Any]: First partition of splitted data.
                - DataFrame[Any]: Second partition of splitted data.
        """
        data_prep, idx_col = self._prepare_stable_data(data)

        # Deterministic sorting
        if timestamp_label in data.columns:
            data_prep = data_prep.sort(by=[user_id_label, timestamp_label, idx_col])
            rank_target_col = timestamp_label
        else:
            data_prep = data_prep.sort(by=[user_id_label, idx_col])
            rank_target_col = idx_col

        # Pre-filtering
        data_prep = data_prep.with_columns(n_items=nw.len().over(user_id_label))
        data_prep = data_prep.filter(nw.col("n_items") > k)

        # Descending sorting
        data_prep = data_prep.with_columns(
            rank_desc=nw.col(rank_target_col)
            .rank(method="ordinal", descending=True)
            .over(user_id_label)
        )

        # Masking
        test_mask = nw.col("rank_desc") <= k

        # Clean the output
        cols_to_drop = ["n_items", "rank_desc", idx_col]
        train = data_prep.filter(~test_mask).drop(cols_to_drop)
        test = data_prep.filter(test_mask).drop(cols_to_drop)

        return [(train, test)]


@splitting_registry.register(SplittingStrategies.TIMESTAMP_SLICING)
class TimestampSlicingSplit(SplittingStrategy):
    """Splits data based on a timestamp. Either a fixed timestamp or a
    'best' timestamp can be used.

    In case of best timestamp strategy, the slicing will be conducted finding the
    timestamp that better separates the transactions. Using the normal fixed strategy
    some users might be completely cut out of the train or the test. With the best
    strategy we ensure that the most amount of users will be represented in both sets.

    For further details about the 'best' timestamp,
        check the `paper <https://link.springer.com/chapter/10.1007/978-3-030-15712-8_63>`_.

    Timestamp must be provided to use this strategy.
    """

    def __call__(
        self,
        data: FrameT,
        user_id_label: str = "user_id",
        timestamp_label: str = "timestamp",
        timestamp: Union[int, str] = 0,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame[Any], DataFrame[Any]]]:
        """Implementation of the fixed timestamp splitting.

        Args:
            data (FrameT): The FrameT to be split.
            user_id_label (str): The user_id label.
            timestamp_label (str): The timestamp label.
            timestamp (Union[int, str]): The timestamp to split data for test set.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame[Any], DataFrame[Any]]]:
                - DataFrame[Any]: First partition of splitted data.
                - DataFrame[Any]: Second partition of splitted data.

        Raises:
            ValueError: If the timestamp column is not present in the FrameT.
        """
        if timestamp_label not in data.columns:
            raise ValueError(
                "The timestamp column must be present inside the data "
                "to apply TimestampSlicing splitting strategy. Timestamp column "
                f"provided: {timestamp_label} not found."
            )

        data_prep, idx_col = self._prepare_stable_data(data)

        if timestamp == "best":
            # Deterministic sorting
            data_prep = data_prep.sort(by=[user_id_label, timestamp_label, idx_col])

            # Find the best timestamp
            best_timestamp = self._best_split(data_prep, user_id_label, timestamp_label)

            # Clean the output
            data_prep = data_prep.drop(idx_col)

            # Apply the split
            first_partition, second_partition = self._fixed_split(
                data_prep, best_timestamp, timestamp_label
            )
        else:
            # Clean the output
            data_prep = data_prep.drop(idx_col)

            # Apply the split
            first_partition, second_partition = self._fixed_split(
                data_prep, int(timestamp), timestamp_label
            )

        return [(first_partition, second_partition)]

    def _best_split(
        self,
        data: DataFrame[Any],
        user_id_label: str,
        timestamp_label: str,
        min_below: int = 1,
        min_over: int = 1,
    ) -> int:
        """Optimized method to find the best split timestamp for partitioning data.

        Args:
            data (DataFrame[Any]): The original data in DataFrame[Any] format.
            user_id_label (str): The user_id label.
            timestamp_label (str): The timestamp label.
            min_below (int): Minimum number of transactions below the timestamp.
            min_over (int): Minimum number of transactions above the timestamp.

        Returns:
            int: Best timestamp for splitting user transactions.
        """
        # Conversion to numpy
        df_dict = data.select(user_id_label, timestamp_label).to_dict(as_series=False)
        u_ids = np.array(df_dict[user_id_label])
        ts_vals = np.array(df_dict[timestamp_label])

        # Create the candidate set
        unique_timestamps = np.unique(np.sort(ts_vals))
        n_candidates = unique_timestamps.shape[0]
        candidate_scores = np.zeros(n_candidates, dtype=int)

        # Search for the optimal timestamp
        user_change_indices = np.where(u_ids[:-1] != u_ids[1:])[0] + 1
        user_splits = np.split(ts_vals, user_change_indices)

        for user_ts in user_splits:
            total_events = user_ts.shape[0]
            below_counts = np.searchsorted(user_ts, unique_timestamps, side="left")
            over_counts = total_events - below_counts

            valid = (below_counts >= min_below) & (over_counts >= min_over)
            candidate_scores += valid.astype(int)

        # Define the timestamp and return it
        max_score = candidate_scores.max()
        valid_candidates = unique_timestamps[candidate_scores == max_score]
        best_timestamp = valid_candidates.max()

        return best_timestamp

    def _fixed_split(
        self, data: DataFrame[Any], timestamp: int, timestamp_label: str = "timestamp"
    ) -> Tuple[DataFrame[Any], DataFrame[Any]]:
        """Method to split data in two partitions, using a fixed timestamp.

        This method will split data based on the timestamp provided.

        Args:
            data (DataFrame[Any]): The original data in DataFrame[Any] format.
            timestamp (int): The timestamp to be used for splitting.
            timestamp_label (str): The timestamp label.

        Returns:
            Tuple[DataFrame[Any], DataFrame[Any]]:
                - DataFrame[Any]: First partition of splitted data.
                - DataFrame[Any]: Second partition of splitted data.
        """
        split_mask = nw.col(timestamp_label) < timestamp
        return data.filter(split_mask), data.filter(~split_mask)


@splitting_registry.register(SplittingStrategies.RANDOM_HOLDOUT)
class RandomHoldoutSplit(SplittingStrategy):
    """The definition of a random ratio splitting strategy."""

    def __call__(
        self,
        data: FrameT,
        ratio: float = 0.2,
        seed: int = 42,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame[Any], DataFrame[Any]]]:
        """Method to split data in two partitions, using a ratio.

        This method will split data based on the ratio provided.

        Args:
            data (FrameT): The original data in FrameT format.
            ratio (float): Percentage of data that will end up in the second partition.
            seed (int): The seed used for the random number generator.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame[Any], DataFrame[Any]]]:
                - DataFrame[Any]: First partition of splitted data.
                - DataFrame[Any]: Second partition of splitted data.
        """
        data_prep, idx_col = self._prepare_stable_data(data)

        # Deterministic sorting before the random part
        data_prep = data_prep.sort(idx_col)

        # Random sampling
        n_rows = data_prep.select(nw.col(data_prep.columns[0])).shape[0]
        np.random.seed(seed)
        random_vals = np.random.rand(n_rows)

        # Assign the random values
        rand_series = nw.new_series(
            name="rand_col",
            values=random_vals,
            backend=nw.get_native_namespace(data_prep),
        )
        data_rand = data_prep.with_columns(rand_col=rand_series)

        # Sort again based on random column
        # NOTE: Using the our indexing as a tie-breaker
        data_rand = data_rand.sort("rand_col", idx_col)

        # Positional split
        data_rand = data_rand.with_row_index(name="__pos_idx__")
        split_idx = int(n_rows * (1 - ratio))

        # Clean the output
        cols_to_drop = ["rand_col", "__pos_idx__", idx_col]
        train = data_rand.filter(nw.col("__pos_idx__") < split_idx).drop(cols_to_drop)
        test = data_rand.filter(nw.col("__pos_idx__") >= split_idx).drop(cols_to_drop)

        return [(train, test)]


@splitting_registry.register(SplittingStrategies.RANDOM_LEAVE_K_OUT)
class RandomLeaveKOutSplit(SplittingStrategy):
    """The definition of a random leave k out splitting strategy."""

    def __call__(
        self,
        data: FrameT,
        user_id_label: str = "user_id",
        k: int = 1,
        seed: int = 42,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame[Any], DataFrame[Any]]]:
        """Method to split data in two partitions, using a ratio.

        This method will split data based on the ratio provided.

        Args:
            data (FrameT): The original data in FrameT format.
            user_id_label (str): The user_id label.
            k (int): Number of transaction that will end up in the second partition.
            seed (int): The seed used for the random number generator.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame[Any], DataFrame[Any]]]:
                - DataFrame[Any]: First partition of splitted data.
                - DataFrame[Any]: Second partition of splitted data.
        """
        data_prep, idx_col = self._prepare_stable_data(data)

        # Pre-filtering
        data_prep = data_prep.with_columns(n_items=nw.len().over(user_id_label))
        data_prep = data_prep.filter(nw.col("n_items") > k)

        # Deterministic sorting before the random part
        data_prep = data_prep.sort(idx_col)

        # Random sampling
        n_rows = data_prep.select(nw.col(data_prep.columns[0])).shape[0]
        np.random.seed(seed)
        random_vals = np.random.rand(n_rows)

        # Assign the random values
        rand_series = nw.new_series(
            name="rand_col",
            values=random_vals,
            backend=nw.get_native_namespace(data_prep),
        )
        data_prep = data_prep.with_columns(rand_col=rand_series)

        # Sort again based on random column
        # NOTE: Using the our indexing as a tie-breaker
        data_prep = data_prep.sort(by=[user_id_label, "rand_col", idx_col])
        data_prep = data_prep.with_columns(
            rnd_rank=nw.col("rand_col").rank(method="ordinal").over(user_id_label)
        )

        # Masking
        test_mask = nw.col("rnd_rank") <= k

        # Clean the output
        cols_to_drop = ["n_items", "rand_col", "rnd_rank", idx_col]
        train = data_prep.filter(~test_mask).drop(cols_to_drop)
        test = data_prep.filter(test_mask).drop(cols_to_drop)

        return [(train, test)]


@splitting_registry.register(SplittingStrategies.K_FOLD_CROSS_VALIDATION)
class KFoldCrossValidation(SplittingStrategy):
    """The definition of KFold Cross Validation."""

    def __call__(
        self,
        data: FrameT,
        folds: int,
        user_id_label: str = "user_id",
        **kwargs: Any,
    ) -> List[Tuple[DataFrame[Any], DataFrame[Any]]]:
        """Method to split data in 'folds' times.

        Args:
            data (FrameT): The original data in FrameT format.
            folds (int): The number of folds to create.
            user_id_label (str): The user_id label.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame[Any], DataFrame[Any]]]:
                - DataFrame[Any]: First partition of splitted data.
                - DataFrame[Any]: Second partition of splitted data.
        """
        data_prep, idx_col = self._prepare_stable_data(data)

        # Deterministic sorting
        data_prep = data_prep.sort(by=[user_id_label, idx_col])

        # Generate fold index
        data_with_folds = data_prep.with_columns(
            seq_id=nw.col(idx_col).rank(method="ordinal").over(user_id_label)
        )
        data_with_folds = data_with_folds.with_columns(
            fold=(nw.col("seq_id") - 1) % folds
        )

        tuple_list = []
        cols_to_drop = ["seq_id", "fold", idx_col]

        # Iterate through the folds
        for i in range(folds):
            # Filter
            test = data_with_folds.filter(nw.col("fold") == i)
            train = data_with_folds.filter(nw.col("fold") != i)

            # Explicit sorting to ensure the ordering of the tuples
            # NOTE: Some backends might lose the order during the filtering process
            train = train.sort(idx_col).drop(cols_to_drop)
            test = test.sort(idx_col).drop(cols_to_drop)

            tuple_list.append((train, test))

        return tuple_list


class ColdStartSplit(SplittingStrategy):
    """Hold out every interaction of a sampled fraction of one kind of entity.

    The usual strategies hold out *interactions*, leaving every user and item with
    some history to learn from. A cold-start protocol holds out *entities*: the
    sampled ones keep nothing, so at evaluation time the model is asked about
    something it has never seen. That is the only way to measure what a
    content-based or hybrid model is actually for.

    Each subclass names the entity it samples through ``COLD_DIMENSION``, which is
    what the splitter and the dataset read to know which side must survive.
    """

    def _split_on(
        self,
        data: FrameT,
        label: str,
        ratio: float,
        seed: int,
    ) -> List[Tuple[DataFrame[Any], DataFrame[Any]]]:
        """Hold out every row belonging to a sampled fraction of one column.

        Args:
            data (FrameT): The data to split.
            label (str): The column naming the entity to sample.
            ratio (float): The fraction of *entities* held out, not of rows. A
                cold-start split is described by how much of the catalogue is
                unseen, so the number of test rows follows from how active those
                entities were.
            seed (int): The seed of the sample.

        Returns:
            List[Tuple[DataFrame[Any], DataFrame[Any]]]: The single train/test pair.

        Raises:
            ValueError: If the ratio would leave nothing on one side of the split.
        """
        if not 0 < ratio < 1:
            raise ValueError(
                f"A cold-start ratio must lie strictly between 0 and 1, got {ratio}."
            )

        data_prep, idx_col = self._prepare_stable_data(data)

        entities = (
            data_prep.select(label).unique().sort(label).to_dict(as_series=False)[label]
        )
        n_cold = int(round(len(entities) * ratio))
        if n_cold == 0 or n_cold == len(entities):
            raise ValueError(
                f"A ratio of {ratio} over {len(entities)} distinct '{label}' values "
                f"holds out {n_cold} of them, which leaves one side of the split "
                "empty. Choose a ratio the catalogue can support."
            )

        rng = np.random.default_rng(seed)
        cold = set(
            np.asarray(entities, dtype=object)[
                rng.choice(len(entities), size=n_cold, replace=False)
            ].tolist()
        )

        is_cold = nw.col(label).is_in(list(cold))
        train = data_prep.filter(~is_cold).drop([idx_col])
        test = data_prep.filter(is_cold).drop([idx_col])

        return [(train, test)]


@splitting_registry.register(SplittingStrategies.ITEM_COLD_START)
class ItemColdStartSplit(ColdStartSplit):
    """Hold out every interaction of a sampled fraction of the items."""

    COLD_DIMENSION: Optional[str] = "item"

    def __call__(
        self,
        data: FrameT,
        item_id_label: str = "item_id",
        ratio: float = 0.1,
        seed: int = 42,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame[Any], DataFrame[Any]]]:
        """Split the data so that a fraction of the items is unseen in training.

        Args:
            data (FrameT): The DataFrame to be splitted.
            item_id_label (str): The item_id label.
            ratio (float): The fraction of items held out.
            seed (int): The seed used for the sample.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame[Any], DataFrame[Any]]]: The train and test sets.
        """
        return self._split_on(data, item_id_label, ratio, seed)


@splitting_registry.register(SplittingStrategies.USER_COLD_START)
class UserColdStartSplit(ColdStartSplit):
    """Hold out every interaction of a sampled fraction of the users."""

    COLD_DIMENSION: Optional[str] = "user"

    def __call__(
        self,
        data: FrameT,
        user_id_label: str = "user_id",
        ratio: float = 0.1,
        seed: int = 42,
        **kwargs: Any,
    ) -> List[Tuple[DataFrame[Any], DataFrame[Any]]]:
        """Split the data so that a fraction of the users is unseen in training.

        Args:
            data (FrameT): The DataFrame to be splitted.
            user_id_label (str): The user_id label.
            ratio (float): The fraction of users held out.
            seed (int): The seed used for the sample.
            **kwargs (Any): The additional keyword arguments.

        Returns:
            List[Tuple[DataFrame[Any], DataFrame[Any]]]: The train and test sets.
        """
        return self._split_on(data, user_id_label, ratio, seed)
