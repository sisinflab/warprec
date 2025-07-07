from typing import Tuple, Any, Optional
from abc import ABC, abstractmethod

import torch
import numpy as np
from torch import Tensor
from pandas import DataFrame
from warprec.data.dataset import Interactions, Sessions
from warprec.utils.enums import RatingType
from warprec.utils.logger import logger


class Dataset(ABC):
    """Abstract base class for Dataset object.

    This class defines a common interface for all Datasets.

    Attributes:
        train_set (Interactions): Training set on that will be used with recommendation models.
        test_set (Interactions): Test set, not mandatory, used in evaluation to calculate metrics.
        val_set (Interactions): Validation set, not mandatory,
            used during training to validate the process.
        train_session (Sessions): Training session used by sequential models.
        test_session (Sessions): Test session, not mandatory used by sequential models.
        val_session (Sessions): Training session, not mandatory used by sequential models.
        user_cluster (Optional[dict]): User cluster information.
        item_cluster (Optional[dict]): Item cluster information.
    """

    train_set: Interactions = None
    test_set: Interactions = None
    val_set: Interactions = None
    train_session: Sessions = None
    test_session: Sessions = None
    val_session: Sessions = None
    user_cluster: Optional[dict] = None
    item_cluster: Optional[dict] = None

    def __init__(self):
        # Set mappings
        self._has_explicit_ratings: bool = False
        self._has_timestamp: bool = False
        self._nuid: int = 0
        self._niid: int = 0
        self._nfeat: int = 0
        self._max_seq_len: int = 0
        self._umap: dict[Any, int] = {}
        self._imap: dict[Any, int] = {}
        self._uc: Tensor = None
        self._ic: Tensor = None

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

    def get_user_cluster(self) -> Tensor:
        """This method retrieves the lookup tensor for user clusters.

        Returns:
            Tensor: Lookup tensor for user clusters.
        """
        return self._uc

    def get_item_cluster(self) -> Tensor:
        """This method retrieves the lookup tensor for item clusters.

        Returns:
            Tensor: Lookup tensor for item clusters.
        """
        return self._ic

    def info(self) -> dict:
        """This method returns the main information of the
        dataset in dict format.

        Returns:
            dict: The dictionary with the main information of
                the dataset.
        """
        return {
            "has_explicit_ratings": self._has_explicit_ratings,
            "has_timestamp": self._has_timestamp,
            "items": self._niid,
            "users": self._nuid,
            "features": self._nfeat,
            "item_mapping": self._imap,
            "user_mapping": self._umap,
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
        user_cluster (Optional[DataFrame]): The user cluster data.
        item_cluster (Optional[DataFrame]): The item cluster data.
        batch_size (int): The batch size that will be used in training and evaluation.
        rating_type (RatingType): The type of rating used.
        rating_label (str): The label of the rating column.
        timestamp_label (str): The label of the timestamp column.
        cluster_label (str): The label of the cluster column.
        need_session_based_information (bool): Wether or not to initialize session data.
        precision (Any): The precision of the internal representation of the data.
    """

    def __init__(
        self,
        train_data: DataFrame,
        test_data: Optional[DataFrame] = None,
        val_data: Optional[DataFrame] = None,
        side_data: Optional[DataFrame] = None,
        user_cluster: Optional[DataFrame] = None,
        item_cluster: Optional[DataFrame] = None,
        batch_size: int = 1024,
        rating_type: RatingType = RatingType.IMPLICIT,
        rating_label: str = None,
        timestamp_label: str = None,
        cluster_label: str = None,
        need_session_based_information: bool = False,
        precision: Any = np.float32,
    ):
        super().__init__()
        # Set user and item label
        user_label = train_data.columns[0]
        item_label = train_data.columns[1]

        # If side information data has been provided, we filter the main dataset
        if side_data is not None:
            train_data, test_data, val_data = self._filter_data(
                train=train_data,
                filter_data=side_data,
                label=item_label,
                test=test_data,
                val=val_data,
            )

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

        # Save other information about the data
        self._has_explicit_ratings = (
            True if rating_type == RatingType.EXPLICIT else False
        )
        self._has_timestamp = (
            True
            if self._has_explicit_ratings
            and len(train_data.columns) == 4
            or not self._has_explicit_ratings
            and len(train_data.columns) == 3
            else False
        )

        # Save side information inside the dataset
        self.side = side_data if side_data is not None else None

        # Save user and item cluster information inside the dataset
        self.user_cluster = (
            {
                self._umap[user_id]: cluster
                for user_id, cluster in zip(
                    user_cluster[user_label], user_cluster[cluster_label]
                )
                if user_id in self._umap
            }
            if user_cluster is not None
            else None
        )
        self.item_cluster = (
            {
                self._imap[item_id]: cluster
                for item_id, cluster in zip(
                    item_cluster[item_label], item_cluster[cluster_label]
                )
                if item_id in self._imap
            }
            if item_cluster is not None
            else None
        )

        # Pre compute lookup tensors for user clusters
        if self.user_cluster is not None:
            unique_user_clusters = sorted(set(self.user_cluster.values()))
            user_cluster_remap = {
                cud: idx + 1 for idx, cud in enumerate(unique_user_clusters)
            }  # Use appropriate indexes for clusters
            self._uc = torch.zeros(self._nuid, dtype=torch.long)
            for u, c in self.user_cluster.items():
                self._uc[u] = user_cluster_remap[c]
        else:
            self._uc = torch.ones(self._nuid, dtype=torch.long)

        # Pre compute lookup tensors for item clusters
        if self.item_cluster is not None:
            unique_item_clusters = sorted(set(self.item_cluster.values()))
            item_cluster_remap = {
                cid: idx + 1 for idx, cid in enumerate(unique_item_clusters)
            }  # Use appropriate indexes for clusters
            self._ic = torch.zeros(self._niid, dtype=torch.long)
            for i, c in self.item_cluster.items():
                self._ic[i] = item_cluster_remap[c]
        else:
            self._ic = torch.ones(self._niid, dtype=torch.long)

        # Create the main data structures
        self.train_set = self._create_inner_set(
            train_data,
            side_data=side_data,
            user_cluster=self.user_cluster,
            item_cluster=self.item_cluster,
            header_msg="Train",
            batch_size=batch_size,
            rating_type=rating_type,
            rating_label=rating_label,
            timestamp_label=timestamp_label,
            precision=precision,
        )

        if test_data is not None:
            self.test_set = self._create_inner_set(
                test_data,
                side_data=side_data,
                user_cluster=self.user_cluster,
                item_cluster=self.item_cluster,
                header_msg="Test",
                batch_size=batch_size,
                rating_type=rating_type,
                rating_label=rating_label,
                timestamp_label=timestamp_label,
                precision=precision,
            )
        if val_data is not None:
            self.val_set = self._create_inner_set(
                val_data,
                side_data=side_data,
                user_cluster=self.user_cluster,
                item_cluster=self.item_cluster,
                header_msg="Validation",
                batch_size=batch_size,
                rating_type=rating_type,
                rating_label=rating_label,
                timestamp_label=timestamp_label,
                precision=precision,
            )

        # Sequential recommendation sessions
        if need_session_based_information:
            self.train_session = Sessions(
                train_data,
                self._umap,
                self._imap,
                batch_size=batch_size,
                timestamp_label=timestamp_label,
            )
            if test_data is not None:
                self.test_session = Sessions(
                    test_data,
                    self._umap,
                    self._imap,
                    batch_size=batch_size,
                    timestamp_label=timestamp_label,
                )
            if val_data is not None:
                self.val_session = Sessions(
                    val_data,
                    self._umap,
                    self._imap,
                    batch_size=batch_size,
                    timestamp_label=timestamp_label,
                )

    def _filter_data(
        self,
        train: DataFrame,
        filter_data: DataFrame,
        label: str,
        test: Optional[DataFrame],
        val: Optional[DataFrame],
    ) -> Tuple[DataFrame, Optional[DataFrame], Optional[DataFrame]]:
        """Filter the data based on a given additional information set and label.

        Args:
            train (DataFrame): The main dataset.
            filter_data (DataFrame): The additional information dataset.
            label (str): The label used to filter the data.
            test (Optional[DataFrame]): The test dataset.
            val (Optional[DataFrame]): The validation dataset.

        Returns:
            Tuple[DataFrame, Optional[DataFrame], Optional[DataFrame]]:
                - DataFrame: The filtered train dataset.
                - Optional[DataFrame]: The filtered test dataset.
                - Optional[DataFrame]: The filtered validation dataset.
        """
        # Compute shared data points first
        shared_data = set(train[label]).intersection(filter_data[label])

        # Count the number of data points before filtering
        train_data_before_filter = train[label].nunique()

        # Filter all the data based on data points present in both train data and filter.
        # This procedure is fundamental because we need dimensions to match
        train = train[train[label].isin(shared_data)]
        filter_data = filter_data[filter_data[label].isin(shared_data)]

        # Check the optional data and also filter them
        if test is not None:
            test = test[test[label].isin(shared_data)]

        if val is not None:
            val = val[val[label].isin(shared_data)]

        # Count the number of data points after filtering
        train_data_after_filter = train[label].nunique()

        logger.attention(
            ""
            f"Filtered out {train_data_before_filter - train_data_after_filter} {label}."
        )

        return train, test, val

    def _create_inner_set(
        self,
        data: DataFrame,
        side_data: Optional[DataFrame] = None,
        user_cluster: Optional[dict] = None,
        item_cluster: Optional[dict] = None,
        header_msg: str = "Train",
        batch_size: int = 1024,
        rating_type: RatingType = RatingType.IMPLICIT,
        rating_label: str = None,
        timestamp_label: str = None,
        precision: Any = np.float32,
    ) -> Interactions:
        """Functionality to create Interaction data from DataFrame.

        Args:
            data (DataFrame): The data used to create the interaction object.
            side_data (Optional[DataFrame]): The side data information about the dataset.
            user_cluster (Optional[dict]): The user cluster information.
            item_cluster (Optional[dict]): The item cluster information.
            header_msg (str): The header of the logger output.
            batch_size (int): The batch size of the interaction.
            rating_type (RatingType): The type of rating used.
            rating_label (str): The label of the rating column.
            timestamp_label (str): The label of the timestamp column.
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
            user_cluster=user_cluster,
            item_cluster=item_cluster,
            batch_size=batch_size,
            rating_type=rating_type,
            rating_label=rating_label,
            timestamp_label=timestamp_label,
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
