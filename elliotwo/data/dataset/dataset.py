from abc import ABC, abstractmethod

from typing import Tuple, Any
from elliotwo.utils.config import Configuration
from elliotwo.data.dataset import Interactions


class AbstractDataset(ABC):
    """Abstract base class for Dataset object.

    This class defines a common interface for all Datasets.

    Attributes:
        train_set (Interactions): Training set on that will be used with recommendation models.
        val_set (Interactions): Validation set, not mandatory, \
            used during training to validate the process.
        test_set (Interactions): Test set, not mandatory, used in evaluation to calculate metrics.

    Args:
        config (Configuration): The configuration file.
    """

    train_set: Interactions
    val_set: Interactions
    test_set: Interactions

    def __init__(self, config: Configuration):
        self._config = config

        # Get information from config file
        self._user_label = self._config.data.labels.user_id_label
        self._item_label = self._config.data.labels.item_id_label

        # Set mappings
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
        self._umap = user_mapping
        self._imap = item_mapping


class TransactionDataset(AbstractDataset):
    """The definition of the Dataset class that will handle transaction data.

    Args:
        train_set (Interactions): The training set.
        user_mapping (dict): The mapping of user ID -> user idx.
        item_mapping (dict): The mapping of item ID -> item idx.
        nuid (int): Number of user IDs.
        niid (int): Number of item IDs.
        config (Configuration): The configuration file.
        val_set (Interactions): The validation set.
        test_set (Interactions): The test set.
    """

    def __init__(
        self,
        train_set: Interactions,
        user_mapping: dict,
        item_mapping: dict,
        nuid: int,
        niid: int,
        config: Configuration,
        val_set: Interactions = None,
        test_set: Interactions = None,
    ):
        super().__init__(config)

        # Set the datasets
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self._nuid = nuid
        self._niid = niid
        self._umap = user_mapping
        self._imap = item_mapping

    def get_dims(self) -> Tuple[int, int]:
        return (self._nuid, self._niid)

    def get_mappings(self) -> Tuple[dict, dict]:
        return (self._umap, self._imap)

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
