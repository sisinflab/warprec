from pandas import DataFrame
from elliotwo.data.dataset import Interactions
from elliotwo.utils.config import Configuration
from elliotwo.data.dataset import TransactionDataset, ContextDataset
from elliotwo.utils.enums import RatingType, SplittingStrategies
from elliotwo.utils.registry import splitting_registry
from elliotwo.utils.logger import logger


class Splitter:
    """Splitter class will handle the splitting of the data.

    Args:
        config (Configuration): Configuration file.

    Attributes:
        read_from_config (bool): Flag to check if the configuration file is being read.
    """

    read_from_config: bool = False

    def __init__(self, config: Configuration = None):
        if config:
            self.read_from_config = True
            self.strategy = config.splitter.strategy
            self._batch_size = config.data.batch_size
            self._seed = config.splitter.seed
            self._user_label = config.data.labels.user_id_label
            self._item_label = config.data.labels.item_id_label
            self._time_label = config.data.labels.timestamp_label
            self._rating_label = config.data.labels.rating_label
            self._rating_type = config.data.rating_type
            self._validation = config.splitter.validation
            self._test_size = (
                config.splitter.ratio[1] if config.splitter.ratio else None
            )
            self._val_size = config.splitter.ratio[2] if config.splitter.ratio else None
            self._test_k = config.splitter.k[0] if config.splitter.k else None
            self._val_k = config.splitter.k[1] if config.splitter.k else None

    def split_transaction(
        self,
        data: DataFrame,
        strategy: SplittingStrategies = None,
        batch_size: int = 1024,
        seed: int = 42,
        user_label: str = "user_id",
        item_label: str = "item_id",
        time_label: str = "timestamp",
        rating_label: str = "rating",
        rating_type: RatingType = RatingType.IMPLICIT,
        validation: bool = False,
        test_size: float = 0.2,
        val_size: float | None = None,
        test_k: int = 1,
        val_k: int | None = None,
    ) -> TransactionDataset:
        """The main method of the class. This method must be called to split the data.

        When called, this method will return the splitting calculated by
        the splitting method selected in the configuration file.

        This method accepts transaction data, and will return a TransactionDataset object.

        A transaction is defined by at least a user_id, an item_id.

        Args:
            data (DataFrame): The DataFrame to be splitted.
            strategy (SplittingStrategies): The splitting strategy to use.
            batch_size (int): The batch size to use for the Interactions objects.
            seed (int): The seed used during splitting.
            user_label (str): The user label in the DataFrame.
            item_label (str): The item label in the DataFrame.
            time_label (str): The timestamp label in the DataFrame.
            rating_label (str): The rating label in the DataFrame.
            rating_type (RatingType): The rating type.
            validation (bool): Wether or not to produce also a validation set.
            test_size (float): The test set size.
            val_size (float | None): The validation set size.
            test_k (int): The k value for test set.
            val_k (int | None): The k value for validation set.

        Returns:
            TransactionDataset: The dataset for the experiment.
        """
        # Initialize the variables to be used
        _strategy = self.strategy if self.read_from_config else strategy
        _batch_size = self._batch_size if self.read_from_config else batch_size
        _seed = self._seed if self.read_from_config else seed
        _user_label = self._user_label if self.read_from_config else user_label
        _item_label = self._item_label if self.read_from_config else item_label
        _time_label = self._time_label if self.read_from_config else time_label
        _rating_label = self._rating_label if self.read_from_config else rating_label
        _rating_type = self._rating_type if self.read_from_config else rating_type
        _validation = self._validation if self.read_from_config else validation
        _test_size = self._test_size if self.read_from_config else test_size
        _val_size = self._val_size if self.read_from_config else val_size
        _test_k = self._test_k if self.read_from_config else test_k
        _val_k = self._val_k if self.read_from_config else val_k

        logger.msg(
            f"Starting splitting process with {_strategy.value} splitting strategy."
        )

        # Get indexes using chosen strategy
        splitter = splitting_registry.get(_strategy)
        idxs = splitter.split(
            data=data,
            seed=_seed,
            user_label=_user_label,
            item_label=_item_label,
            time_label=_time_label,
            test_size=_test_size,
            val_size=_val_size,
            test_k=_test_k,
            val_k=_val_k,
        )

        # Define train/val/test subset of _inter_df taking into account
        # only user and items present in train set.
        _train_set = data.iloc[idxs[0]]
        if _validation:
            _val_set = data.iloc[idxs[1]]
            _val_set = _val_set[
                _val_set[_user_label].isin(_train_set[_user_label])
                & _val_set[_item_label].isin(_train_set[_item_label])
            ]
        _test_set = data.iloc[idxs[2]]
        _test_set = _test_set[
            _test_set[_user_label].isin(_train_set[_user_label])
            & _test_set[_item_label].isin(_train_set[_item_label])
        ]

        # Define dimensions that will lead the experiment
        _nuid = _train_set[_user_label].nunique()
        _niid = _train_set[_item_label].nunique()

        # Update mappings inside Dataset structure
        _uid = _train_set[_user_label].unique()
        _iid = _train_set[_item_label].unique()

        # Calculate mapping for users and items
        _umap = {user: i for i, user in enumerate(_uid)}
        _imap = {item: i for i, item in enumerate(_iid)}

        # Create splits inside Dataset class for future use
        train_set = Interactions(
            _train_set,
            (_nuid, _niid),
            _umap,
            _imap,
            batch_size=_batch_size,
            user_id_label=_user_label,
            item_id_label=_item_label,
            rating_label=_rating_label,
            rating_type=_rating_type,
        )
        if _validation:
            val_set = Interactions(
                _val_set,
                (_nuid, _niid),
                _umap,
                _imap,
                batch_size=_batch_size,
                user_id_label=_user_label,
                item_id_label=_item_label,
                rating_label=_rating_label,
                rating_type=_rating_type,
            )
        else:
            val_set = None
        test_set = Interactions(
            _test_set,
            (_nuid, _niid),
            _umap,
            _imap,
            batch_size=_batch_size,
            user_id_label=_user_label,
            item_id_label=_item_label,
            rating_label=_rating_label,
            rating_type=_rating_type,
        )

        # Logging of splitting process
        logger.msg("Splitting process over.")

        # Train set stats
        train_nuid, train_niid = train_set.get_dims()
        train_transactions = train_set.get_transactions()
        logger.stat_msg(
            (
                f"Number of users: {train_nuid}      "
                f"Number of items: {train_niid}      "
                f"Transactions: {train_transactions}"
            ),
            "Train set",
        )
        test_nuid, test_niid = test_set.get_dims()
        test_transactions = test_set.get_transactions()
        logger.stat_msg(
            (
                f"Number of users: {test_nuid}      "
                f"Number of items: {test_niid}      "
                f"Transactions: {test_transactions}"
            ),
            "Test set",
        )
        if _validation:
            val_nuid, val_niid = val_set.get_dims()
            val_transactions = val_set.get_transactions()
            logger.stat_msg(
                (
                    f"Number of users: {val_nuid}      "
                    f"Number of items: {val_niid}      "
                    f"Transactions: {val_transactions}"
                ),
                "Validation set",
            )

        return TransactionDataset(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            user_mapping=_umap,
            item_mapping=_imap,
            nuid=_nuid,
            niid=_niid,
        )

    def split_context(self, data: DataFrame) -> ContextDataset:
        """This function will be used to split context data."""
        raise NotImplementedError
