import typing
from typing import Tuple, Any, Optional

import torch
import numpy as np
from pandas import DataFrame
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import csr_matrix, coo_matrix
from warprec.utils.enums import RatingType


class Interactions:
    """Interactions class will handle the data of the transactions.

    Args:
        data (DataFrame): Transaction data in DataFrame format.
        original_dims (Tuple[int, int]):
            int: Number of users.
            int: Number of items.
        user_mapping (dict): Mapping of user ID -> user idx.
        item_mapping (dict): Mapping of item ID -> item idx.
        side_data (Optional[DataFrame]): The side information features in DataFrame format.
        batch_size (int): The batch size that will be used to
            iterate over the interactions.
        rating_type (RatingType): The type of rating to be used.
        precision (Any): The precision of the internal representation of the data.

    Raises:
        ValueError: If the rating type is not supported.
    """

    _inter_dict: dict = {}
    _inter_df: DataFrame = None
    _inter_sparse: csr_matrix = None
    _inter_side_sparse: csr_matrix = None

    def __init__(
        self,
        data: DataFrame,
        original_dims: Tuple[int, int],
        user_mapping: dict,
        item_mapping: dict,
        side_data: Optional[DataFrame] = None,
        batch_size: int = 1024,
        rating_type: RatingType = RatingType.IMPLICIT,
        precision: Any = np.float32,
    ) -> None:
        # Setup the variables
        self._inter_df = data
        self._inter_side = side_data if side_data is not None else None
        self.batch_size = batch_size
        self.rating_type = rating_type
        self.precision = precision

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

    def get_sparse(self) -> csr_matrix:
        """This method retrieves the sparse representation of data.

        This method also checks if the sparse structure has
        already been created, if not then it also create it first.

        Returns:
            csr_matrix: Sparse representation of the transactions (CSR Format).
        """
        if isinstance(self._inter_sparse, csr_matrix):
            return self._inter_sparse
        return self._to_sparse()

    def get_side_sparse(self) -> csr_matrix:
        """This method retrieves the sparse representation of side data.

        This method also checks if the sparse structure has
        already been created, if not then it also create it first.

        Returns:
            csr_matrix: Sparse representation of the features (CSR Format).
        """
        if isinstance(self._inter_side_sparse, csr_matrix):
            return self._inter_side_sparse
        self._inter_side_sparse = csr_matrix(
            self._inter_side.drop(self._item_label, axis=1), dtype=self.precision
        )
        return self._inter_side_sparse

    @typing.no_type_check
    def get_item_rating_dataloader(
        self, num_negatives: int = 0, shuffle: bool = True
    ) -> DataLoader:
        """Create a PyTorch DataLoader with implicit feedback and negative sampling.

        Args:
            num_negatives (int): Number of negative samples per user.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: Yields (user, item, rating) with negative samples.
        """
        # Define main variables
        sparse_matrix = self.get_sparse().tocoo()
        users_pos = sparse_matrix.row
        items_pos = sparse_matrix.col
        num_items = self._niid
        num_positives = len(users_pos)

        # Create positive interactions tensor (implicit feedback)
        pos_users = torch.LongTensor(users_pos)
        pos_items = torch.LongTensor(items_pos)
        pos_ratings = torch.ones_like(pos_users, dtype=torch.float)

        # Precompute positive for each user
        user_pos_items = {
            u: set(items_pos[users_pos == u]) for u in np.unique(users_pos)
        }
        all_items = np.arange(num_items)

        # Preallocate arrays for negatives
        neg_users = np.empty(num_positives * num_negatives, dtype=np.int64)
        neg_items = np.empty_like(neg_users)

        start_idx = 0
        for u, pos_set in user_pos_items.items():
            # For each user, find the items they have not interacted with
            # and sample negative items
            user_mask = users_pos == u
            user_count = np.sum(user_mask)
            neg_candidates = np.setdiff1d(all_items, list(pos_set))

            # Sample all negatives for this user at once
            replace = len(neg_candidates) < num_negatives * user_count
            sampled = np.random.choice(
                neg_candidates, size=num_negatives * user_count, replace=replace
            )

            # Fill preallocated arrays
            end_idx = start_idx + num_negatives * user_count
            neg_users[start_idx:end_idx] = u
            neg_items[start_idx:end_idx] = sampled
            start_idx = end_idx

        # Create negative interactions tensor
        # Note: PyTorch tensors require long integers for indices
        neg_users = torch.LongTensor(neg_users)
        neg_items = torch.LongTensor(neg_items)
        neg_ratings = torch.zeros_like(neg_users, dtype=torch.float)

        # Combine positive and negative samples
        all_users = torch.cat([pos_users, neg_users])
        all_items = torch.cat([pos_items, neg_items])
        all_ratings = torch.cat([pos_ratings, neg_ratings])

        # Create final dataset
        dataset = TensorDataset(all_users, all_items, all_ratings)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    @typing.no_type_check
    def get_pos_neg_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Create a PyTorch DataLoader with triplets for implicit feedback.

        Args:
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: Yields triplets of (user, positive_item, negative_item).
        """
        # Define main variables
        sparse_matrix = self.get_sparse().tocoo()
        users_pos = sparse_matrix.row
        items_pos = sparse_matrix.col
        num_items = self._niid
        num_positives = len(users_pos)

        # Precompute positive items for each user
        user_pos_items = {u: items_pos[users_pos == u] for u in np.unique(users_pos)}
        all_items = np.arange(num_items)

        # Preallocate arrays for triplets
        users_triplet = np.empty(num_positives, dtype=np.int64)
        positives_triplet = np.empty_like(users_triplet)
        negatives_triplet = np.empty_like(users_triplet)

        start_idx = 0
        for u, pos_items in user_pos_items.items():
            user_count = len(pos_items)
            if user_count == 0:  # Skip the user if it has 0 interactions
                continue

            neg_candidates = np.setdiff1d(all_items, pos_items)
            if len(neg_candidates) == 0:  # Extreme edge case. Normally not a problem
                continue

            # Sample negatives for all positives in this user
            replace = len(neg_candidates) < user_count
            sampled = np.random.choice(neg_candidates, size=user_count, replace=replace)

            # Create triplet components
            end_idx = start_idx + user_count
            users_triplet[start_idx:end_idx] = np.full(user_count, u)
            positives_triplet[start_idx:end_idx] = pos_items
            negatives_triplet[start_idx:end_idx] = sampled
            start_idx = end_idx

        # Trim to actual size and convert to tensors
        # Note: PyTorch tensors require long integers for indices
        users_tensor = torch.LongTensor(users_triplet[:start_idx])
        positives_tensor = torch.LongTensor(positives_triplet[:start_idx])
        negatives_tensor = torch.LongTensor(negatives_triplet[:start_idx])

        # Create final dataset
        dataset = TensorDataset(users_tensor, positives_tensor, negatives_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

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
        users = self._inter_df[self._user_label].map(self._umap).values
        items = self._inter_df[self._item_label].map(self._imap).values
        ratings = (
            self._inter_df[
                self._rating_label
            ].values  # With explicit rating we read from dictionary
            if self.rating_type == RatingType.EXPLICIT
            else np.ones(
                self._transactions
            )  # With implicit rating we create an array directly (faster)
        )
        self._inter_sparse = coo_matrix(
            (ratings, (users, items)),
            shape=(self._og_nuid, self._og_niid),
            dtype=self.precision,
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
