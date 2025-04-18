from typing import Tuple, Any, Optional
from abc import ABC, abstractmethod

import numpy as np
from pandas import DataFrame
from elliotwo.data.dataset import Interactions
from elliotwo.utils.enums import RatingType
from elliotwo.utils.logger import logger


class Dataset(ABC):
    """Abstract base class for Dataset object.

    This class defines a common interface for all Datasets.

    Attributes:
        train_set (Interactions): Training set on that will be used with recommendation models.
        val_set (Interactions): Validation set, not mandatory,
            used during training to validate the process.
        test_set (Interactions): Test set, not mandatory, used in evaluation to calculate metrics.
    """

    train_set: Interactions = None
    val_set: Interactions = None
    test_set: Interactions = None

    def __init__(self):
        # Set mappings
        self._nuid: int = 0
        self._niid: int = 0
        self._nfeat: int = 0
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

    def info(self) -> dict:
        """This method returns the main information of the
        dataset in dict format.

        Returns:
            dict: The dictionary with the main information of
                the dataset.
        """
        return {
            "items": self._niid,
            "users": self._nuid,
            "features": self._nfeat,
        }

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

            return train_batch, test_batch, val_batch
        except StopIteration as exc:
            raise exc


class TransactionDataset(Dataset):
    """The definition of the Dataset class that will handle transaction data.

    Args:
        train_data (DataFrame): The train data.
        test_data (Optional[DataFrame]): The test data.
        val_data (Optional[DataFrame]): The validation data.
        side_data (Optional[DataFrame]): The side information data.
        batch_size (int): The batch size that will be used in training and evaluation.
        rating_type (RatingType): The type of rating used in the dataset.
        precision (Any): The precision of the internal representation of the data.
    """

    def __init__(
        self,
        train_data: DataFrame,
        test_data: Optional[DataFrame] = None,
        val_data: Optional[DataFrame] = None,
        side_data: Optional[DataFrame] = None,
        batch_size: int = 1024,
        rating_type: RatingType = RatingType.IMPLICIT,
        precision: Any = np.float32,
    ):
        super().__init__()
        # Set user and item label
        user_label = train_data.columns[0]
        item_label = train_data.columns[1]

        # If side information data has been provided, we filter the main dataset
        if side_data is not None:
            train_data = train_data[train_data[item_label].isin(side_data[item_label])]
            test_data = (
                test_data[test_data[item_label].isin(side_data[item_label])]
                if test_data is not None
                else None
            )
            val_data = (
                val_data[val_data[item_label].isin(side_data[item_label])]
                if val_data is not None
                else None
            )
            side_data = side_data[side_data[item_label].isin(train_data[item_label])]

        # Define dimensions that will lead the experiment
        self._nuid = train_data[user_label].nunique()
        self._niid = train_data[item_label].nunique()
        self._nfeat = len(side_data.columns) - 1 if side_data is not None else 0

        # Values that will be used to calculate mappings
        _uid = train_data[user_label].unique()
        _iid = train_data[item_label].unique()

        # Calculate mapping for users and items
        self._umap = {user: i for i, user in enumerate(_uid)}
        self._imap = {item: i for i, item in enumerate(_iid)}

        # Save side information inside the dataset
        self.side = side_data if side_data is not None else None

        # Create the main data structures
        self.train_set = self._create_inner_set(
            train_data,
            side_data=side_data,
            header_msg="Train",
            batch_size=batch_size,
            rating_type=rating_type,
            precision=precision,
        )

        if test_data is not None:
            self.test_set = self._create_inner_set(
                test_data,
                side_data=side_data,
                header_msg="Test",
                batch_size=batch_size,
                rating_type=rating_type,
                precision=precision,
            )
        if val_data is not None:
            self.val_set = self._create_inner_set(
                val_data,
                side_data=side_data,
                header_msg="Validation",
                batch_size=batch_size,
                rating_type=rating_type,
                precision=precision,
            )

    def _create_inner_set(
        self,
        data: DataFrame,
        side_data: Optional[DataFrame] = None,
        header_msg: str = "Train",
        batch_size: int = 1024,
        rating_type: RatingType = RatingType.IMPLICIT,
        precision: Any = np.float32,
    ) -> Interactions:
        """Functionality to create Interaction data from DataFrame.

        Args:
            data (DataFrame): The data used to create the interaction object.
            side_data (Optional[DataFrame]): The side data information about the dataset.
            header_msg (str): The header of the logger output.
            batch_size (int): The batch size of the interaction.
            rating_type (RatingType): The type of rating used.
            precision (Any): The precision that will be used to store interactions.

        Returns:
            Interactions: The final interaction object.
        """
        inter_set = Interactions(
            data,
            (self._nuid, self._niid),
            self._umap,
            self._imap,
            side_data=side_data,
            batch_size=batch_size,
            rating_type=rating_type,
            precision=precision,
        )
        nuid, niid = inter_set.get_dims()
        transactions = inter_set.get_transactions()
        logger.stat_msg(
            (
                f"Number of users: {nuid}      "
                f"Number of items: {niid}      "
                f"Transactions: {transactions}"
            ),
            f"{header_msg} set",
        )

        return inter_set

    def get_dims(self) -> Tuple[int, int]:
        return (self._nuid, self._niid)

    def get_mappings(self) -> Tuple[dict, dict]:
        return (self.umap, self.imap)

    def get_inverse_mappings(self) -> Tuple[dict, dict]:
        return {v: k for k, v in self._umap.items()}, {
            v: k for k, v in self._imap.items()
        }


class ContextDataset(Dataset):
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
