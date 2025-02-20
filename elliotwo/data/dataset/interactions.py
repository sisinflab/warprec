from typing import Tuple

import numpy as np
from pandas import DataFrame
from scipy.sparse import csr_matrix, coo_matrix
from elliotwo.utils.enums import RatingType
from elliotwo.utils.config import Configuration


class Interactions:
    """Interactions class will handle the data of the transactions.

    Args:
        data (DataFrame): Transaction data in DataFrame format.
        config (Configuration): Configuration file.
        original_dims (Tuple[int, int]):
            int: Number of users.
            int: Number of items.
        user_mapping (dict): Mapping of user ID -> user idx.
        item_mapping (dict): Mapping of item ID -> item idx.

    Raises:
        ValueError: If the rating type is not supported.
    """

    def __init__(
        self,
        data: DataFrame,
        config: Configuration,
        original_dims: Tuple[int, int],
        user_mapping: dict,
        item_mapping: dict,
    ) -> None:
        self._inter_dict = {}
        self._inter_df = data
        self._index = 0
        self._inter_sparse = None
        self._config = config

        # Retrieve informations from config
        self._rating_type = self._config.data.rating_type
        self._user_label = self._config.data.labels.user_id_label
        self._item_label = self._config.data.labels.item_id_label
        self._score_label = self._config.data.labels.rating_label

        # Definition of important attributes
        self._uid = self._inter_df[self._user_label].unique()
        self._nuid = self._inter_df[self._user_label].nunique()
        self._niid = self._inter_df[self._item_label].nunique()
        self._og_nuid, self._og_niid = original_dims
        self._transactions = len(self._inter_df)
        self._umap = user_mapping
        self._imap = item_mapping

        # Define the interaction dictionary, based on the RatingType selected
        if self._rating_type == RatingType.EXPLICIT:
            for u in self._uid:
                _user_interactions = self._inter_df[
                    self._inter_df[self._user_label] == u
                ]
                self._inter_dict[u] = dict(
                    zip(
                        _user_interactions[self._item_label],
                        _user_interactions[self._score_label],
                    )
                )
        elif self._rating_type == RatingType.IMPLICIT:
            for u in self._uid:
                _user_interactions = self._inter_df[
                    self._inter_df[self._user_label] == u
                ]
                _user_int_len = len(_user_interactions)
                self._inter_dict[u] = dict(
                    zip(
                        _user_interactions[self._item_label].tolist(),
                        np.ones(_user_int_len).tolist(),
                    )
                )
        else:
            raise ValueError(f"Rating type {self._rating_type} not supported.")

    def get_dict(self) -> dict:
        """This method will return the transaction information in dict format.

        Returns:
            dict: The transaction information in the current \
                representation {user ID: {item ID: rating}}.
        """
        return self._inter_dict

    def get_df(self) -> DataFrame:
        """This method will return the raw data.

        Returns:
            DataFrame: The raw data in tabular format.
        """
        return self._inter_df

    def get_sparse(self) -> csr_matrix:
        """This method retrieves the sparse representation of data.

        This method also checks if the sparse structure has \
            already been created, if not then it also create it first.

        Returns:
            csr_matrix: Sparse representation of the transactions (CSR Format).
        """
        if isinstance(self._inter_sparse, csr_matrix):
            return self._inter_sparse
        return self._to_sparse()

    def get_dims(self) -> Tuple[int, int]:
        """This method will return the dimensions of the data.

        Returns:
            Tuple[int, int]: A tuple containing:
                int: Number of unique users.
                int: Number of unique items.
        """
        return (self._nuid, self._niid)

    def get_transactions(self) -> int:
        """This method will return the number of transactions.

        Returns:
            int: Number of transactions.
        """
        return self._transactions

    def _to_sparse(self) -> csr_matrix:
        """This method will create the sparse representation of the data contained.

        This method must not be called if the sparse representation has already be defined.

        Returns:
            csr_matrix: Sparse representation of the transactions (CSR Format).
        """
        users = []
        items = []
        ratings = []

        # Iter self and get all tuple of interactions
        for u, i, r in self:
            users.append(self._umap[u])
            items.append(self._imap[i])
            ratings.append(r)

        # Create sparse structure
        self._inter_sparse = coo_matrix(
            (ratings, (users, items)),
            shape=(self._og_nuid, self._og_niid),
            dtype=self._config.precision_numpy(),
        ).tocsr()
        return self._inter_sparse

    def __iter__(self) -> "Interactions":
        """This method will return the iterator of the interactions.

        Returns:
            Interactions: The iterator of the interactions.
        """
        self._index = 0
        return self

    def __next__(self) -> Tuple[int, int, float]:
        """This method will return the next interaction in the data.

        Returns:
            Tuple[int, int, float]:
                int: User ID.
                int: Item ID.
                float: Rating.
        Raises:
            StopIteration: If the end of the data is reached.
        """
        if self._index < len(self._inter_df):
            result = self._inter_df.iloc[self._index]
            self._index += 1
            return (
                result[self._user_label],
                result[self._item_label],
                result[self._score_label],
            )
        raise StopIteration

    def __len__(self) -> int:
        """This method calculates the length of the interactions.

        Length will be defined as the number of ratings.

        Returns:
            int: number of ratings present in the structure.
        """
        return sum(len(ir) for _, ir in self._inter_dict.items())
