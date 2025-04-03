from typing import Tuple, Any, Optional
from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame
from elliotwo.data.dataset import Interactions
from elliotwo.utils.enums import RatingType
from elliotwo.utils.logger import logger


class AbstractDataset(ABC):
    """Abstract base class for Dataset object.

    This class defines a common interface for all Datasets.

    Attributes:
        train_set (Interactions): Training set on that will be used with recommendation models.
        val_set (Interactions): Validation set, not mandatory,
            used during training to validate the process.
        test_set (Interactions): Test set, not mandatory, used in evaluation to calculate metrics.
    """

    train_set: Interactions
    val_set: Interactions
    test_set: Interactions

    def __init__(self):
        # Set mappings
        self._nuid: int = 0
        self._niid: int = 0
        self._umap: dict[Any, int] = {}
        self._imap: dict[Any, int] = {}

    @abstractmethod
    def get_dims(self) -> Tuple[int, int]:
        """Returns the dimensions of the data.

        Returns:
            Tuple[int, int]:
                int: Number of unique user_ids.
                int: Number of unique item_ids.
        """

    @abstractmethod
    def get_mappings(self) -> Tuple[dict, dict]:
        """Returns the mapping used for this dataset.

        Returns:
            Tuple[dict, dict]:
                dict: Mapping of user_id -> user_idx.
                dict: Mapping of item_id -> item_idx.
        """

    @abstractmethod
    def get_inverse_mappings(self) -> Tuple[dict, dict]:
        """Returns the inverse of the mapping.

        Returns:
            Tuple[dict, dict]:
                dict: Mapping of user_idx -> user_id.
                dict: Mapping of item_idxs -> item_id.
        """

    def update_mappings(self, user_mapping: dict, item_mapping: dict):
        """Update the mappings of the dataset.

        Args:
            user_mapping (dict): The mapping of user_id -> user_idx.
            item_mapping (dict): The mapping of item_id -> item_idx.
        """
        self.umap = user_mapping
        self.imap = item_mapping

    def __iter__(self):
        self.train_iter = iter(self.train_set)
        self.val_iter = iter(self.val_set) if self.val_set else None
        self.test_iter = iter(self.test_set) if self.test_set else None

        return self

    def __next__(self):
        try:
            train_batch = next(self.train_iter)
            val_batch = next(self.val_iter) if self.val_iter else None
            test_batch = next(self.test_iter) if self.test_iter else None

            return train_batch, val_batch, test_batch
        except StopIteration as exc:
            raise exc


class TransactionDataset(AbstractDataset):
    """The definition of the Dataset class that will handle transaction data.

    Args:
        train_data (DataFrame): The train data.
        test_data (Optional[DataFrame]): The test data.
        val_data (Optional[DataFrame]): The validation data.
        batch_size (int): The batch size that will be used in training and evaluation.
        rating_type (RatingType): The type of rating used in the dataset.
        precision (Any): The precision of the internal representation of the data.
    """

    def __init__(
        self,
        train_data: DataFrame,
        test_data: Optional[DataFrame] = None,
        val_data: Optional[DataFrame] = None,
        batch_size: int = 1024,
        rating_type: RatingType = RatingType.IMPLICIT,
        precision: Any = np.float32,
    ):
        super().__init__()
        # Set user and item label
        user_label = train_data.columns[0]
        item_label = train_data.columns[1]

        # Define dimensions that will lead the experiment
        self._nuid = train_data[user_label].nunique()
        self._niid = train_data[item_label].nunique()

        # Update mappings inside Dataset structure
        _uid = train_data[user_label].unique()
        _iid = train_data[item_label].unique()

        # Calculate mapping for users and items
        self._umap = {user: i for i, user in enumerate(_uid)}
        self._imap = {item: i for i, item in enumerate(_iid)}

        self.train_set = Interactions(
            train_data,
            (self._nuid, self._niid),
            self._umap,
            self._imap,
            batch_size=batch_size,
            rating_type=rating_type,
            precision=precision,
        )

        # Train set stats
        train_nuid, train_niid = self.train_set.get_dims()
        train_transactions = self.train_set.get_transactions()
        logger.stat_msg(
            (
                f"Number of users: {train_nuid}      "
                f"Number of items: {train_niid}      "
                f"Transactions: {train_transactions}"
            ),
            "Train set",
        )

        if test_data is not None:
            self.test_set = Interactions(
                test_data,
                (self._nuid, self._niid),
                self._umap,
                self._imap,
                batch_size=batch_size,
                rating_type=rating_type,
                precision=precision,
            )
            test_nuid, test_niid = self.test_set.get_dims()
            test_transactions = self.test_set.get_transactions()
            logger.stat_msg(
                (
                    f"Number of users: {test_nuid}      "
                    f"Number of items: {test_niid}      "
                    f"Transactions: {test_transactions}"
                ),
                "Test set",
            )
        if val_data is not None:
            self.val_set = Interactions(
                val_data,
                (self._nuid, self._niid),
                self._umap,
                self._imap,
                batch_size=batch_size,
                rating_type=rating_type,
                precision=precision,
            )
            val_nuid, val_niid = self.val_set.get_dims()
            val_transactions = self.val_set.get_transactions()
            logger.stat_msg(
                (
                    f"Number of users: {val_nuid}      "
                    f"Number of items: {val_niid}      "
                    f"Transactions: {val_transactions}"
                ),
                "Validation set",
            )

    def get_dims(self) -> Tuple[int, int]:
        return (self._nuid, self._niid)

    def get_mappings(self) -> Tuple[dict, dict]:
        return (self.umap, self.imap)

    def get_inverse_mappings(self) -> Tuple[dict, dict]:
        return {v: k for k, v in self._umap.items()}, {
            v: k for k, v in self._imap.items()
        }


class ContextDataset(AbstractDataset):
    """The definition of the Dataset class that will handle context-aware data.

    TODO: Implement
    """

    @abstractmethod
    def get_dims(self) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def get_mappings(self) -> Tuple[dict, dict]:
        raise NotImplementedError

    @abstractmethod
    def get_inverse_mappings(self) -> Tuple[dict, dict]:
        raise NotImplementedError
