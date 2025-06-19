import typing
from typing import Tuple, Any, Optional

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from pandas import DataFrame
from torch.utils.data import DataLoader, TensorDataset, Dataset
from scipy.sparse import csr_matrix, coo_matrix
from warprec.utils.enums import RatingType
from warprec.utils.logger import logger


class SequentialInteractions(Dataset):
    """Personalized dataset for sequential data.

    Used by sequential models to capture temporal information from
    user interaction history.

    Args:
        sequences (Tensor): The sequence of interactions.
        sequence_lengths (Tensor): The length of the sequence (before padding).
        positive_items (Tensor): The positive items.
        negative_items (Optional[Tensor]): The negative items.
    """

    def __init__(
        self,
        sequences: Tensor,
        sequence_lengths: Tensor,
        positive_items: Tensor,
        negative_items: Optional[Tensor] = None,
    ):
        self.sequences = sequences
        self.sequence_lengths = sequence_lengths
        self.positive_items = positive_items
        self.negative_items = negative_items

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.negative_items is not None:
            return (
                self.sequences[idx],
                self.sequence_lengths[idx],
                self.positive_items[idx],
                self.negative_items[idx],
            )
        return (
            self.sequences[idx],
            self.sequence_lengths[idx],
            self.positive_items[idx],
        )


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
        user_cluster (Optional[dict]): The user cluster information.
        item_cluster (Optional[dict]): The item cluster information.
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
    _history_matrix: Tensor = None
    _history_lens: Tensor = None
    _history_mask: Tensor = None
    _cached_sequential_data: dict = {}

    def __init__(
        self,
        data: DataFrame,
        original_dims: Tuple[int, int],
        user_mapping: dict,
        item_mapping: dict,
        side_data: Optional[DataFrame] = None,
        user_cluster: Optional[dict] = None,
        item_cluster: Optional[dict] = None,
        batch_size: int = 1024,
        rating_type: RatingType = RatingType.IMPLICIT,
        precision: Any = np.float32,
    ) -> None:
        # Setup the variables
        self._inter_df = data
        self._inter_side = side_data.copy() if side_data is not None else None
        self._inter_user_cluster = user_cluster if user_cluster is not None else None
        self._inter_item_cluster = item_cluster if item_cluster is not None else None
        self.batch_size = batch_size
        self.rating_type = rating_type
        self.precision = precision

        # Set user and item label
        self._user_label = data.columns[0]
        self._item_label = data.columns[1]
        self._rating_label = (
            data.columns[2] if rating_type == RatingType.EXPLICIT else None
        )

        # Filter side information (if present)
        if self._inter_side is not None:
            self._inter_side = self._inter_side[
                self._inter_side[self._item_label].isin(
                    self._inter_df[self._item_label]
                )
            ]

            # Order side information to be in the same order of the dataset
            self._inter_side["order"] = self._inter_side[self._item_label].map(
                item_mapping
            )
            self._inter_side = (
                self._inter_side.sort_values("order")
                .drop(columns="order")
                .reset_index(drop=True)
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
        if self._inter_side is None:
            return None
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
        sparse_matrix = self.get_sparse()
        sparse_matrix_coo = sparse_matrix.tocoo()
        num_users = self._nuid
        num_items = self._niid

        # Single call of module for efficiency
        rint = np.random.randint  # Storing the module call is ever so slightly faster

        # Create positive interactions tensor (implicit feedback)
        pos_users = torch.LongTensor(sparse_matrix_coo.row)
        pos_items = torch.LongTensor(sparse_matrix_coo.col)
        num_positives = len(pos_users)

        # Create tensors of positives
        pos_users_tensor = torch.LongTensor(pos_users)
        pos_items_tensor = torch.LongTensor(pos_items)
        pos_ratings_tensor = torch.ones(num_positives, dtype=torch.float)

        if num_negatives == 0:  # Check if negative samples are required
            if num_positives == 0:  # Edge case: No interactions
                dataset = TensorDataset(
                    torch.LongTensor([]), torch.LongTensor([]), torch.FloatTensor([])
                )  # Empty dataset
            else:
                dataset = TensorDataset(
                    pos_users_tensor, pos_items_tensor, pos_ratings_tensor
                )  # Only positive dataset
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        # Other edge case: No positive interactions
        elif num_positives == 0:
            dataset = TensorDataset(
                torch.LongTensor([]), torch.LongTensor([]), torch.FloatTensor([])
            )
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        total_samples = (
            num_positives * num_negatives
        )  # At this point this will be a positive number

        neg_users = np.empty(total_samples, dtype=np.int64)
        neg_items = np.empty(total_samples, dtype=np.int64)

        current_neg_idx = 0
        for u in range(num_users):
            # Using sparse CSR matrix, get the indices of nnz columns
            # these will be the positive items
            start_ptr = sparse_matrix.indptr[u]
            end_ptr = sparse_matrix.indptr[u + 1]

            # Get indices of items interacted (positive items)
            user_pos = sparse_matrix.indices[start_ptr:end_ptr]
            user_count = len(user_pos)
            if user_count == 0:  # Skip the user if it has 0 interactions
                continue

            # Number of user negative samples
            user_neg = user_count * num_negatives
            user_pos_set = set(user_pos)  # Efficient control using sets

            # Edge case: the user interacted with all the items
            if user_count == num_items:
                continue  # Skip the user if it interacted with all items

            # Iter for each negative samples for this user
            for _ in range(user_neg):
                # Until we find a valid negative, keep searching
                while True:
                    candidate_neg_item = rint(0, num_items)
                    if candidate_neg_item not in user_pos_set:
                        # If found save and break loop
                        neg_users[current_neg_idx] = u
                        neg_items[current_neg_idx] = candidate_neg_item
                        current_neg_idx += 1
                        break

        # Trim length based on possible triplets skipped
        neg_users_trimmed = neg_users[:current_neg_idx]
        neg_items_trimmed = neg_items[:current_neg_idx]

        # Create Tensors for efficient data loading
        neg_users_tensor = torch.LongTensor(neg_users_trimmed)
        neg_items_tensor = torch.LongTensor(neg_items_trimmed)
        neg_ratings_tensor = torch.zeros(current_neg_idx, dtype=torch.float)

        # Concatenate complete tensors for final dataset
        all_users = torch.cat([pos_users_tensor, neg_users_tensor])
        all_items = torch.cat([pos_items_tensor, neg_items_tensor])
        all_ratings = torch.cat([pos_ratings_tensor, neg_ratings_tensor])

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
        sparse_matrix = self.get_sparse()
        num_users = self._nuid
        num_items = self._niid
        num_positives = sparse_matrix.nnz

        # Preallocate arrays for triplets
        users_triplet = np.empty(num_positives, dtype=np.int64)
        positives_triplet = np.empty_like(users_triplet)
        negatives_triplet = np.empty_like(users_triplet)

        # Single call of module for efficiency
        rint = np.random.randint  # Storing the module call is ever so slightly faster

        current_idx = 0
        for u in range(num_users):
            # Using sparse CSR matrix, get the indices of nnz columns
            # these will be the positive items
            start_ptr = sparse_matrix.indptr[u]
            end_ptr = sparse_matrix.indptr[u + 1]

            # Get indices of items interacted (positive items)
            user_pos = sparse_matrix.indices[start_ptr:end_ptr]
            user_count = len(user_pos)
            if user_count == 0:  # Skip the user if it has 0 interactions
                continue

            user_pos_set = set(user_pos)  # Efficient control using sets

            # Edge case: the user interacted with all the items
            if user_count == num_items:
                continue  # Skip the user if it interacted with all items

            # Iter through all the positive items
            for pos_item in user_pos:
                # Assign user and positive item to arrays
                users_triplet[current_idx] = u
                positives_triplet[current_idx] = pos_item

                # Until we find a valid negative, keep searching
                while True:
                    candidate_neg_item = rint(0, num_items)

                    if (
                        candidate_neg_item not in user_pos_set
                    ):  # If found save and break loop
                        negatives_triplet[current_idx] = candidate_neg_item
                        break

                current_idx += 1

        # Trim length based on possible triplets skipped
        users_triplet_trimmed = users_triplet[:current_idx]
        positives_triplet_trimmed = positives_triplet[:current_idx]
        negatives_triplet_trimmed = negatives_triplet[:current_idx]

        # Create Tensors for efficient data loading
        users_tensor = torch.LongTensor(users_triplet_trimmed)
        positives_tensor = torch.LongTensor(positives_triplet_trimmed)
        negatives_tensor = torch.LongTensor(negatives_triplet_trimmed)

        # Create final dataset
        dataset = TensorDataset(users_tensor, positives_tensor, negatives_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def get_history(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the history representation as three Tensors.

        This method also checks if this representation has been already computed,
        if so then it just returns it without computing it again.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - Tensor: A matrix of dimension [num_user, max_chronology_length],
                    containing transaction information.
                - Tensor: An array of dimension [num_user], containing the
                    length of each chronology (before padding).
                - Tensor: A binary mask that identifies where the real
                    transaction information are, ignoring padding.
        """
        if (
            isinstance(self._history_matrix, Tensor)
            and isinstance(self._history_lens, Tensor)
            and isinstance(self._history_mask, Tensor)
        ):
            return self._history_matrix, self._history_lens, self._history_mask
        return self._to_history()

    # Your original method, now calling the optimized helper function
    def get_sequential_dataloader(
        self, num_negatives: int = 0, shuffle: bool = True
    ) -> DataLoader:
        """Create a dataloader for sequential data.

        Args:
            num_negatives (int): Number of negative samples per user.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: Yields (item_seq, item_seq_len, pos_item_id) if num_negatives = 0.
                Yields (item_seq, item_seq_len, pos_item_id, neg_item_id) if num_negatives > 0.
        """
        # Check if sequential that has been cached
        cache_key = num_negatives
        if cache_key in self._cached_sequential_data:
            tensors = self._cached_sequential_data[cache_key]
            dataset = SequentialInteractions(*[t for t in tensors if t is not None])
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        # Call the optimized processing function
        padded_item_seq, tensor_item_seq_len, tensor_pos_item_id, tensor_neg_item_id = (
            self._create_sequences_and_targets(
                num_negatives=num_negatives,
            )
        )

        # Check for empty results
        if padded_item_seq.shape[0] == 0:
            logger.attention(
                "No valid sequential samples generated for DataLoader. "
                "Check your data or session definition (min length 2)."
            )

        # Cache the results
        self._cached_sequential_data[cache_key] = (
            padded_item_seq,
            tensor_item_seq_len,
            tensor_pos_item_id,
            tensor_neg_item_id,
        )

        # Create final dataset and dataloader
        tensors_for_dataset = [padded_item_seq, tensor_item_seq_len, tensor_pos_item_id]
        if num_negatives > 0 and tensor_neg_item_id is not None:
            tensors_for_dataset.append(tensor_neg_item_id)

        dataset = SequentialInteractions(*tensors_for_dataset)
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

    def _to_history(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Creates three Tensor which contains information of the
        transaction history for each user.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - Tensor: A matrix of dimension [num_user, max_chronology_length],
                    containing transaction information.
                - Tensor: An array of dimension [num_user], containing the
                    length of each chronology (before padding).
                - Tensor: A binary mask that identifies where the real
                    transaction information are, ignoring padding.
        """
        # Get sparse interaction matrix
        sparse_matrix = self.get_sparse()

        # Get user_ids and item_ids from interactions
        user_ids, item_ids = sparse_matrix.nonzero()

        # Create a dictionary to store history for each user
        user_history_dict: dict[int, list] = {}
        for u, i in zip(user_ids, item_ids):
            if u not in user_history_dict:
                user_history_dict[u] = []
            user_history_dict[u].append(
                i + 1
            )  # Add 1 to item_id to reserve 0 for padding

        # Determine max history length for padding
        max_history_len = 0
        for user_id in range(self._nuid):
            if user_id in user_history_dict:
                max_history_len = max(max_history_len, len(user_history_dict[user_id]))

        # Initialize matrices
        self._history_matrix = torch.zeros(
            self._nuid, max_history_len, dtype=torch.long
        )
        self._history_lens = torch.zeros(self._nuid, dtype=torch.long)
        self._history_mask = torch.zeros(self._nuid, max_history_len, dtype=torch.float)

        # Populate matrices
        for user_id in range(self._nuid):
            if user_id in user_history_dict:
                items = torch.tensor(user_history_dict[user_id], dtype=torch.long)
                self._history_matrix[user_id, : len(items)] = items
                self._history_lens[user_id] = len(items)
                self._history_mask[user_id, : len(items)] = 1.0

        return self._history_matrix, self._history_lens, self._history_mask

    def _create_sequences_and_targets(
        self,
        num_negatives: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Core logic for transforming interaction data into sequential training samples.

        This function uses pandas and numpy for efficient processing.

        Args:
            num_negatives (int): Number of negative samples per user.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]: A tuple containing:
                - Tensor: Padded item sequence.
                - Tensor: Item sequence length.
                - Tensor: Positive item tensor.
                - Optional[Tensor]: Negative item tensor.
        """
        mapped_df = pd.DataFrame(
            {
                self._user_label: self._inter_df[self._user_label].map(self._umap),
                self._item_label: self._inter_df[self._item_label].map(self._imap),
                "timestamp": self._inter_df["timestamp"],
            }
        ).dropna()  # Drop any interactions if user/item not in map

        # Convert to integer types
        mapped_df[[self._user_label, self._item_label]] = mapped_df[
            [self._user_label, self._item_label]
        ].astype(int)

        # Group interactions based on timestamp
        user_sessions = (
            mapped_df.sort_values(by=[self._user_label, "timestamp"])
            .groupby(self._user_label)[self._item_label]
            .agg(list)
        )

        # Filter out sessions with less than 2 interactions
        user_sessions = user_sessions[user_sessions.str.len() >= 2]

        # Edge case: No user has at least 2 interactions in sequence
        if user_sessions.empty:
            return (
                torch.empty((0, 0), dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                None if num_negatives == 0 else torch.empty(0, dtype=torch.long),
            )

        # Create sequence-target pairs using explode
        # For a session [i1, i2, i3, i4], this generates:
        #   - seq: [i1], target: i2
        #   - seq: [i1, i2], target: i3
        #   - seq: [i1, i2, i3], target: i4
        session_df = pd.DataFrame(
            {
                "sequences": user_sessions.apply(
                    lambda s: [s[: i + 1] for i in range(len(s) - 1)]
                ),
                "targets": user_sessions.apply(lambda s: s[1:]),
            }
        )

        # Explode the lists into separate rows
        training_data = session_df.explode(["sequences", "targets"]).reset_index()

        # Handle Negative Sampling if requested
        neg_tensor = None
        if num_negatives > 0:
            num_samples = len(training_data)

            # Create a set of interacted items for each sequence for fast lookups
            # Note: items are 0-indexed here
            interacted_sets = training_data["sequences"].apply(set)

            # Generate all negative candidates at once
            neg_candidates = np.random.randint(
                0, self._niid, size=(num_samples, num_negatives), dtype=np.int64
            )

            # Check for collisions: a negative sample is invalid if it's the target or in the history
            is_target = neg_candidates == training_data["targets"].values[:, None]

            # We iterate through rows and columns to build a boolean mask
            in_history = np.zeros_like(neg_candidates, dtype=bool)
            for i, interacted_set in enumerate(interacted_sets):
                for j in range(num_negatives):
                    if neg_candidates[i, j] in interacted_set:
                        in_history[i, j] = True

            invalid_mask = is_target | in_history

            # Iteratively replace invalid samples until all are valid
            while np.any(invalid_mask):
                num_invalid = np.sum(invalid_mask)
                new_candidates = np.random.randint(
                    0, self._niid, size=num_invalid, dtype=np.int64
                )
                neg_candidates[invalid_mask] = new_candidates

                # Re-evaluate the mask only for the newly generated candidates
                is_target_new = (
                    neg_candidates == training_data["targets"].values[:, None]
                )
                in_history_new = np.zeros_like(neg_candidates, dtype=bool)
                rows, cols = np.where(
                    invalid_mask
                )  # Get indices of what needs checking
                for i, j in zip(rows, cols):  # type: ignore[assignment]
                    if neg_candidates[i, j] in interacted_sets.iloc[i]:
                        in_history_new[i, j] = True

                invalid_mask = is_target_new | in_history_new

            # Convert to 1-indexed for the model
            neg_tensor = torch.tensor(neg_candidates + 1, dtype=torch.long)

        # Convert DataFrame to list
        sequences_list_0_indexed = training_data["sequences"].tolist()

        # Convert list to be 1-indexed, this is faster
        # than working with DataFrames
        sequences_tensors_1_indexed = [
            torch.tensor(s, dtype=torch.long) + 1 for s in sequences_list_0_indexed
        ]

        # Use torch.nn.utils to pad the sequence
        padded_item_seq = torch.nn.utils.rnn.pad_sequence(
            sequences_tensors_1_indexed, batch_first=True, padding_value=0
        )

        # Convert to 1-index and to Tensors
        tensor_item_seq_len = torch.tensor(
            training_data["sequences"].str.len().tolist(), dtype=torch.long
        )
        tensor_pos_item_id = torch.tensor(
            training_data["targets"].values.astype(np.int64) + 1, dtype=torch.long
        )

        return padded_item_seq, tensor_item_seq_len, tensor_pos_item_id, neg_tensor

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
