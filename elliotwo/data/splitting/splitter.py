from pandas import DataFrame
from elliotwo.data.dataset import Interactions
from elliotwo.utils.config import Configuration
from elliotwo.data.dataset import TransactionDataset, ContextDataset
from elliotwo.utils.registry import splitting_registry
from elliotwo.utils.logger import logger


class Splitter:
    """Splitter class will handle the splitting of the data.

    Args:
        config (Configuration): Configuration file.
    """

    def __init__(self, config: Configuration):
        self._config = config
        self._splitting_strategy_name = self._config.splitter.strategy
        self._splitting_strategy = splitting_registry.get(
            self._splitting_strategy_name, config=config
        )

        # Retrieve main labels for later use
        self._user_id = self._config.data.labels.user_id_label
        self._item_id = self._config.data.labels.item_id_label

    def split_transaction(self, data: DataFrame) -> TransactionDataset:
        """The main method of the class. This method must be called to split the data.

        When called, this method will return the splitting calculated by
        the splitting method selected in the configuration file.

        This method accepts transaction data, and will return a TransactionDataset object.

        A transaction is defined by at least a user_id, an item_id.

        Args:
            data (DataFrame): The DataFrame to be splitted.

        Returns:
            TransactionDataset: The dataset for the experiment.
        """
        logger.msg(
            f"Starting splitting process with {self._splitting_strategy_name.value} splitting strategy."
        )

        # Get indexes using chosen strategy
        idxs = self._splitting_strategy.split(data)

        # Define train/val/test subset of _inter_df taking into account
        # only user and items present in train set.
        _train_set = data.iloc[idxs[0]]
        if self._config.splitter.validation:
            _val_set = data.iloc[idxs[1]]
            _val_set = _val_set[
                _val_set[self._user_id].isin(_train_set[self._user_id])
                & _val_set[self._item_id].isin(_train_set[self._item_id])
            ]
        _test_set = data.iloc[idxs[2]]
        _test_set = _test_set[
            _test_set[self._user_id].isin(_train_set[self._user_id])
            & _test_set[self._item_id].isin(_train_set[self._item_id])
        ]

        # Define dimensions that will lead the experiment
        _nuid = _train_set[self._user_id].nunique()
        _niid = _train_set[self._item_id].nunique()

        # Update mappings inside Dataset structure
        _uid = _train_set[self._user_id].unique()
        _iid = _train_set[self._item_id].unique()

        # Calculate mapping for users and items
        _umap = {user: i for i, user in enumerate(_uid)}
        _imap = {item: i for i, item in enumerate(_iid)}

        # Create splits inside Dataset class for future use
        train_set = Interactions(_train_set, self._config, (_nuid, _niid), _umap, _imap)
        if self._config.splitter.validation:
            val_set = Interactions(_val_set, self._config, (_nuid, _niid), _umap, _imap)
        else:
            val_set = None
        test_set = Interactions(_test_set, self._config, (_nuid, _niid), _umap, _imap)

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
        if self._config.splitter.validation:
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
            config=self._config,
            user_mapping=_umap,
            item_mapping=_imap,
            nuid=_nuid,
            niid=_niid,
        )

    def split_context(self, data: DataFrame) -> ContextDataset:
        raise NotImplementedError
