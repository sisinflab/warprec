from typing import Tuple, Any

import numpy as np
from pandas import DataFrame
from scipy.sparse import csr_matrix, coo_matrix
from elliotwo.utils.enums import RatingType


class Interactions:
    """Interactions class will handle the data of the transactions.

    Args:
        data (DataFrame): Transaction data in DataFrame format.
        original_dims (Tuple[int, int]):
            int: Number of users.
            int: Number of items.
        user_mapping (dict): Mapping of user ID -> user idx.
        item_mapping (dict): Mapping of item ID -> item idx.
        batch_size (int): The batch size that will be used to
            iterate over the interactions.
        rating_type (RatingType): The type of rating to be used.

    Raises:
        ValueError: If the rating type is not supported.
    """

    _inter_dict: dict = {}
    _inter_df: DataFrame = None
    _inter_sparse: csr_matrix = None

    def __init__(
        self,
        data: DataFrame,
        original_dims: Tuple[int, int],
        user_mapping: dict,
        item_mapping: dict,
        batch_size: int = 1024,
        rating_type: RatingType = RatingType.IMPLICIT,
    ) -> None:
        # Setup the variables
        self._inter_df = data
        self.batch_size = batch_size
        self.rating_type = rating_type

        # Set user and item label
        self._user_label = data.columns[0]
        self._item_label = data.columns[1]
        self._rating_label = (
            data.columns[2] if rating_type == RatingType.EXPLICIT else None
        )

        # Definition of dimensions
        self._uid = self._inter_df[self._user_label].unique()
        self._nuid = self._inter_df[self._user_label].nunique()
        self._niid = self._inter_df[self._item_label].nunique()
        self._og_nuid, self._og_niid = original_dims
        self._transactions = len(self._inter_df)

        # Set mappings
        self._umap = user_mapping
        self._imap = item_mapping

        # Set the index
        self._index = 0

        # Define the interaction dictionary, based on the RatingType selected
        if rating_type == RatingType.EXPLICIT:
            self._inter_dict = (
                self._inter_df.groupby(self._user_label)
                .apply(
                    lambda df: dict(zip(df[self._item_label], df[self._rating_label]))
                )
                .to_dict()
            )
        elif rating_type == RatingType.IMPLICIT:
            self._inter_dict = {
                user: dict.fromkeys(items, 1)
                for user, items in self._inter_df.groupby(self._user_label)[
                    self._item_label
                ]
            }
        else:
            raise ValueError(f"Rating type {rating_type} not supported.")

    def get_dict(self) -> dict:
        """This method will return the transaction information in dict format.

        Returns:
            dict: The transaction information in the current
                representation {user ID: {item ID: rating}}.
        """
        return self._inter_dict

    def get_df(self) -> DataFrame:
        """This method will return the raw data.

        Returns:
            DataFrame: The raw data in tabular format.
        """
        return self._inter_df

    def get_sparse(self, precision: Any = np.float32) -> csr_matrix:
        """This method retrieves the sparse representation of data.

        This method also checks if the sparse structure has
        already been created, if not then it also create it first.

        Args:
            precision (Any): The precision of the sparse matrix.

        Returns:
            csr_matrix: Sparse representation of the transactions (CSR Format).
        """
        if isinstance(self._inter_sparse, csr_matrix):
            return self._inter_sparse
        return self._to_sparse(precision=precision)

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

    def _to_sparse(self, precision: Any = np.float32) -> csr_matrix:
        """This method will create the sparse representation of the data contained.

        This method must not be called if the sparse representation has already be defined.

        Args:
            precision (Any): The precision of the sparse matrix.

        Returns:
            csr_matrix: Sparse representation of the transactions (CSR Format).
        """
        users = self._inter_df[self._user_label].map(self._umap).values
        items = self._inter_df[self._item_label].map(self._imap).values
        ratings = (
            self._inter_df[self._rating_label].values
            if self.rating_type == RatingType.EXPLICIT
            else np.ones(self._transactions)
        )
        self._inter_sparse = coo_matrix(
            (ratings, (users, items)),
            shape=(self._og_nuid, self._og_niid),
            dtype=precision,
        ).tocsr()
        return self._inter_sparse

    def __iter__(self) -> "Interactions":
        """This method will return the iterator of the interactions.

        Returns:
            Interactions: The iterator of the interactions.
        """
        self._index = 0
        if not isinstance(self._inter_sparse, csr_matrix):
            self._to_sparse()
        return self

    def __next__(self) -> csr_matrix:
        """This method will iterate over the sparse data.

        Returns:
            csr_matrix: The csr representation of data.
        Raises:
            StopIteration: If the end of the data is reached.
            ValueError: If the sparse matrix is None.
        """
        if self._index >= self._og_nuid:
            raise StopIteration
        if self._inter_sparse is None:
            raise ValueError("The sparse matrix is None.")

        start = self._index
        end = min(start + self.batch_size, self._og_nuid)
        self._index = end
        return self._inter_sparse[start:end]

    def __len__(self) -> int:
        """This method calculates the length of the interactions.

        Length will be defined as the number of ratings.

        Returns:
            int: number of ratings present in the structure.
        """
        return sum(len(ir) for _, ir in self._inter_dict.items())
