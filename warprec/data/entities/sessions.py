from typing import Optional, Tuple, Dict, List
from functools import partial

import torch
import pandas as pd
import numpy as np
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from warprec.data.entities.train_structures.custom_collate_fn import (
    collate_fn_cloze_mask,
)
from warprec.data.entities.train_structures import (
    SessionDataset,
    LazySessionDataset,
    UserHistoryDataset,
    LazyUserHistoryDataset,
    ClozeMaskDataset,
    LazyClozeMaskDataset,
)
from warprec.utils.logger import logger


class Sessions:
    """Handles session-based data preparation for sequential recommenders.

    This class transforms a DataFrame of user-item interactions into padded
    sequences suitable for training models in PyTorch. It supports negative
    sampling and caching for efficiency.

    Args:
        data (DataFrame): Transaction data.
        user_mapping (dict): Mapping of original user ID -> integer index.
        item_mapping (dict): Mapping of original item ID -> integer index.
        user_id_label (str): The name of the user ID column in the DataFrame.
        item_id_label (str): The name of the item ID column in the DataFrame.
        timestamp_label (str): The name of the timestamp column.
            If provided, interactions will be sorted by this column.

    Raises:
        ValueError: If the user or item label are not found in the DataFrame.
    """

    def __init__(
        self,
        data: DataFrame,
        user_mapping: dict,
        item_mapping: dict,
        user_id_label: str = "user_id",
        item_id_label: str = "item_id",
        timestamp_label: str = "timestamp",
    ):
        # Validate presence of required columns
        if user_id_label not in data.columns:
            raise ValueError(f"User column '{user_id_label}' not found in DataFrame.")
        if item_id_label not in data.columns:
            raise ValueError(f"Item column '{item_id_label}' not found in DataFrame.")

        # Initialize attributes
        self._inter_df = data
        self._user_label = user_id_label
        self._item_label = item_id_label
        self._timestamp_label = timestamp_label
        self._umap = user_mapping
        self._imap = item_mapping
        self._niid = len(self._imap)
        self._processed_df = None

        # Initialize the cache
        self._cached_dataset: Dict[str, Dataset] = {}
        self._cached_user_histories: Dict[int, List[int]] = {}

    def clear_dataset_cache(self):
        """Clears all cached data to free up memory."""
        del self._cached_dataset
        self._cached_dataset = {}

        del self._cached_user_histories
        self._cached_user_histories = {}

    def _get_processed_data(self) -> DataFrame:
        """Centralized method to map, clean, and sort interaction data.

        This method maps user and item IDs to their integer indices,
        drops any rows with missing values (e.g., unseen users/items),
        and sorts the interactions by user and timestamp (if available).
        The result is cached in self._processed_df to avoid redundant processing.

        Returns:
            DataFrame: The processed and sorted interaction data.
        """
        # Return from cache if already processed
        if self._processed_df is not None:
            return self._processed_df

        # Prepare columns for the new DataFrame
        cols_to_map = {
            self._user_label: self._inter_df[self._user_label].map(self._umap),
            self._item_label: self._inter_df[self._item_label].map(self._imap),
        }

        # Remove not mapped users/items
        mapped_df = pd.DataFrame(cols_to_map).dropna()

        # Include timestamp if it's relevant and exists
        has_timestamp = (
            self._timestamp_label and self._timestamp_label in self._inter_df.columns
        )
        if has_timestamp:
            timestamp_col = self._inter_df.loc[mapped_df.index, self._timestamp_label]
            mapped_df[self._timestamp_label] = timestamp_col

        # Create, clean, and cast types
        mapped_df[[self._user_label, self._item_label]] = mapped_df[
            [self._user_label, self._item_label]
        ].astype(np.int64)

        # Sort the data. If timestamp is available, use it for sorting
        if has_timestamp:
            sorted_df = mapped_df.sort_values(
                by=[self._user_label, self._timestamp_label]
            ).reset_index(drop=True)
        else:
            # Fallback sort if no timestamp
            sorted_df = mapped_df.sort_values(by=self._user_label).reset_index(
                drop=True
            )

        # Cache and return
        self._processed_df = sorted_df
        return self._processed_df

    def _get_user_sessions(self) -> pd.Series:
        """Maps, sorts (if timestamp is available), and groups interactions by user.
        This centralized helper method prevents code duplication.

        Returns:
            pd.Series: A Series where the index is user IDs and the values are lists of item
                interactions in chronological order.
        """
        # Get the centralized, processed, and sorted data
        processed_df = self._get_processed_data()

        # Group by user and aggregate item interactions into a list
        return processed_df.groupby(self._user_label)[self._item_label].agg(list)

    def get_sequential_dataloader(
        self,
        max_seq_len: int,
        neg_samples: int = 0,
        include_user_id: bool = False,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        low_memory: bool = False,
    ) -> DataLoader:
        """Creates a DataLoader for sequential data.

        This method generates padded sequences of user interactions
        along with positive and negative samples for training sequential
        recommendation models.

        Args:
            max_seq_len (int): Maximum length of sequences produced.
            neg_samples (int): Number of negative samples per user.
            include_user_id (bool): Whether to include user IDs in the output.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.
            seed (int): Seed for Numpy random number generator for reproducibility.
            low_memory (bool): Whether to create the dataloader with a lazy approach.

        Returns:
            DataLoader: A DataLoader yielding sequential training samples.
        """
        if low_memory:
            sorted_df = self._get_processed_data()

            # Edge case: No valid session
            if sorted_df.empty:
                logger.negative("No valid session found in the data.")
                return DataLoader(torch.utils.data.TensorDataset(torch.empty(0)))

            lazy_dataset = LazySessionDataset(
                sorted_df=sorted_df,
                user_label=self._user_label,
                item_label=self._item_label,
                max_seq_len=max_seq_len,
                neg_samples=neg_samples,
                niid=self._niid,
                include_user_id=include_user_id,
                seed=seed,
            )

            # Edge case: The dataset contains no data
            if len(lazy_dataset) == 0:
                logger.negative("Session dataset is empty. No valid session found.")

            return DataLoader(lazy_dataset, batch_size=batch_size, shuffle=shuffle)

        # Check cache first
        cache_key = (
            f"sequence_len_{max_seq_len}_neg_{neg_samples}_user_{include_user_id}"
        )
        if cache_key in self._cached_dataset:
            dataset = self._cached_dataset[cache_key]
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        (
            tensor_user_id,
            padded_item_seq,
            tensor_item_seq_len,
            tensor_pos_item_id,
            tensor_neg_item_id,
        ) = self._create_sequences_and_targets(
            neg_samples=neg_samples,
            max_seq_len=max_seq_len,
            include_user_id=include_user_id,
            seed=seed,
        )

        # Edge case: no valid samples
        if padded_item_seq.shape[0] == 0:
            logger.attention(
                "No valid sequential samples generated. Ensure sessions have at least 2 interactions."
            )

        # Cache the generated tensors for future use
        dataset_args = {
            "sequences": padded_item_seq,
            "sequence_lengths": tensor_item_seq_len,
            "positive_items": tensor_pos_item_id,
        }
        if tensor_user_id is not None:
            dataset_args["users"] = tensor_user_id
        if tensor_neg_item_id is not None:
            dataset_args["negative_items"] = tensor_neg_item_id

        # Create dataset and DataLoader
        dataset = SessionDataset(**dataset_args)
        self._cached_dataset[cache_key] = dataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _create_sequences_and_targets(
        self,
        neg_samples: int,
        max_seq_len: int,
        include_user_id: bool = False,
        seed: int = 42,
    ) -> Tuple[Optional[Tensor], Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Core logic for transforming interaction data into sequential training samples.

        This function uses pandas and numpy for efficient processing.

        Args:
            neg_samples (int): Number of negative samples per user.
            max_seq_len (int): Maximum length of sequences produced.
            include_user_id (bool): Wether or not to return also the user_id.
            seed (int): Seed for Numpy random number generator for reproducibility.

        Returns:
            Tuple[Optional[Tensor], Tensor, Tensor, Tensor, Optional[Tensor]]: A tuple containing:
                - Optional[Tensor]: User ID.
                - Tensor: Padded item sequence.
                - Tensor: Item sequence length.
                - Tensor: Positive item tensor.
                - Optional[Tensor]: Negative item tensor.
        """
        # Retrieve the processed and sorted DataFrame
        sorted_df = self._get_processed_data()

        # Handle edge case where no data exists after processing
        if sorted_df.empty:
            return (
                None if not include_user_id else torch.empty(0, dtype=torch.long),
                torch.empty((0, max_seq_len), dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                None if neg_samples == 0 else torch.empty(0, dtype=torch.long),
            )

        # Extract numpy arrays for efficient processing
        all_items_0_indexed = sorted_df[self._item_label].values
        all_users = sorted_df[self._user_label].values

        # Identify user sessions by detecting changes in the user ID column
        is_new_session = np.diff(all_users, prepend=-1) != 0
        session_starts_idx = np.where(is_new_session)[0]
        session_lengths = np.diff(session_starts_idx, append=len(all_users))

        # Filter out sessions that are too short to form training samples
        valid_session_mask = session_lengths >= 2
        valid_session_starts = session_starts_idx[valid_session_mask]
        valid_session_lengths = session_lengths[valid_session_mask]

        if len(valid_session_starts) == 0:
            # Handle edge case where no valid sessions exist
            return (
                None if not include_user_id else torch.empty(0, dtype=torch.long),
                torch.empty((0, max_seq_len), dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                None if neg_samples == 0 else torch.empty(0, dtype=torch.long),
            )

        # For each valid session of length L, we generate L-1 training samples.
        # This block calculates the total number of samples and determines the index
        # of the positive target item for each sample
        num_samples_per_session = valid_session_lengths - 1
        num_total_samples = np.sum(num_samples_per_session)

        sequence_index_offsets = np.concatenate(
            [np.arange(n) for n in num_samples_per_session]
        ).astype(np.int64)

        session_indices = np.repeat(
            np.arange(len(valid_session_starts)), num_samples_per_session
        ).astype(np.int64)

        target_indices = (
            valid_session_starts[session_indices] + sequence_index_offsets + 1
        )

        # Calculate sequence start indices and final lengths
        session_start_for_sample = valid_session_starts[session_indices]
        truncation_start_indices = target_indices - max_seq_len
        true_sequence_start_indices = np.maximum(
            session_start_for_sample, truncation_start_indices
        )
        final_sequence_lengths_np = target_indices - true_sequence_start_indices

        # Gather item data to form sequences
        item_offsets = np.arange(max_seq_len, dtype=np.int64)
        indices_to_gather = true_sequence_start_indices[:, None] + item_offsets

        # Clip the indices before indexing the array
        # NOTE: This is important in scenarios where users have very long sessions
        max_valid_index = len(all_items_0_indexed) - 1
        np.clip(indices_to_gather, 0, max_valid_index, out=indices_to_gather)

        # Convert to PyTorch tensors for indexing
        all_items_tensor = torch.from_numpy(all_items_0_indexed).long()
        indices_to_gather_tensor = torch.from_numpy(indices_to_gather).long()

        # Items are gathered using the computed indices
        padded_item_seq = all_items_tensor[indices_to_gather_tensor]

        # Apply right-padding mask
        tensor_item_seq_len = torch.from_numpy(final_sequence_lengths_np).long()
        row_indices = torch.arange(max_seq_len, dtype=torch.long)
        padding_mask_right = row_indices >= tensor_item_seq_len[:, None]
        padded_item_seq.masked_fill_(padding_mask_right, self._niid)

        # If requested, generate `neg_samples` negative samples for each positive sample.
        # This process is vectorized and ensures that negative candidates do not collide
        # with the positive target or any item in the user's history for that sample
        neg_tensor = None
        if neg_samples > 0:
            # Set random seed for reproducibility
            np.random.seed(seed)

            # Generate initial random candidates
            neg_candidates = np.random.randint(
                0, self._niid, size=(num_total_samples, neg_samples), dtype=np.int64
            )

            # Prepare data for collision checks
            target_items_0_indexed = all_items_0_indexed[target_indices]

            # History is the padded sequence, converted back to 0-indexed.
            # Padding (0) is set to -1 to avoid accidental matches with item 0.
            history_0_indexed = padded_item_seq.numpy().copy()
            history_0_indexed[history_0_indexed == self._niid] = -1
            history_tensor = torch.from_numpy(history_0_indexed).long()

            # Iteratively re-sample until no collisions exist
            invalid_mask = np.ones_like(
                neg_candidates, dtype=bool
            )  # Start with all as potentially invalid
            while np.any(invalid_mask):
                # Check for collision with the positive target
                is_target = neg_candidates == target_items_0_indexed[:, None]

                # Check for collision with sequence history using broadcasting
                neg_candidates_tensor = torch.from_numpy(neg_candidates).long()
                collision_matrix = (
                    neg_candidates_tensor[:, :, None] == history_tensor[:, None, :]
                )
                in_history = torch.any(collision_matrix, dim=2).numpy()

                # Combine masks and find where re-sampling is needed
                invalid_mask = is_target | in_history
                if not np.any(invalid_mask):
                    break

                # Re-sample only the invalid candidates
                num_invalid = np.sum(invalid_mask)
                new_candidates = np.random.randint(
                    0, self._niid, size=num_invalid, dtype=np.int64
                )
                neg_candidates[invalid_mask] = new_candidates

            # Convert final valid candidates to 1-based indexing
            neg_tensor = torch.from_numpy(neg_candidates).long()

        # Convert all remaining numpy arrays to PyTorch tensors and return them.
        tensor_pos_item_id = torch.from_numpy(
            all_items_0_indexed[target_indices]
        ).long()

        # Optionally convert user IDs if requested
        tensor_user_id = None
        if include_user_id:
            tensor_user_id = torch.from_numpy(all_users[target_indices]).long()

        return (
            tensor_user_id,
            padded_item_seq,
            tensor_item_seq_len,
            tensor_pos_item_id,
            neg_tensor,
        )

    def get_user_history_sequences(
        self, user_ids: List[int], max_seq_len: int
    ) -> Tuple[Tensor, Tensor]:
        """Retrieves padded historical sequences for evaluation.

        Args:
            user_ids (List[int]): List of user IDs to retrieve histories for.
            max_seq_len (int): Maximum length of the returned sequences.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Tensor: Padded item sequences for the users.
                - Tensor: Lengths of each user's sequence before padding.
        """
        # Ensure user histories are cached
        if not self._cached_user_histories:
            self._compute_cache_user_history()

        # Gather sequences and lengths
        sequences_to_process = []
        lengths_to_process = []
        for uid in user_ids:
            history = self._cached_user_histories.get(uid, [])
            recent_history = history[-max_seq_len:]
            sequences_to_process.append(torch.tensor(recent_history, dtype=torch.long))
            lengths_to_process.append(len(recent_history))

        # Pad sequences and convert lengths to tensor
        padded_sequences = pad_sequence(
            sequences_to_process, batch_first=True, padding_value=self._niid
        )
        sequence_lengths = torch.tensor(lengths_to_process, dtype=torch.long)
        return padded_sequences, sequence_lengths

    def _compute_cache_user_history(self):
        """Computes and caches the complete interaction history for every user."""
        user_sessions = self._get_user_sessions()
        self._cached_user_histories = user_sessions.apply(
            lambda x: np.array(x).tolist()
        ).to_dict()

    def get_user_history_dataloader(
        self,
        max_seq_len: int,
        neg_samples: int,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        low_memory: bool = False,
    ) -> DataLoader:
        """Creates a DataLoader where each item is a user's full history.

        Args:
            max_seq_len (int): Maximum length of the user history sequence.
            neg_samples (int): Number of negative samples for each positive item.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.
            seed (int): Seed for reproducibility of negative sampling.
            low_memory (bool): Whether to create the dataloader with a lazy approach.

        Returns:
            DataLoader: A DataLoader yielding user history sequences with negative samples.
        """
        if low_memory:
            user_sessions = self._get_user_sessions()

            lazy_dataset = LazyUserHistoryDataset(
                user_sessions=user_sessions,
                max_seq_len=max_seq_len,
                neg_samples=neg_samples,
                niid=self._niid,
                seed=seed,
            )

            # Edge case: No valid history
            if len(lazy_dataset) == 0:
                logger.negative(
                    "No valid user history samples generated. Ensure users have at least 2 interactions."
                )

            return DataLoader(lazy_dataset, batch_size=batch_size, shuffle=shuffle)

        # Check cache first
        cache_key = f"history_len_{max_seq_len}_neg_{neg_samples}"
        if cache_key in self._cached_dataset:
            dataset = self._cached_dataset[cache_key]
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Generate sequences and negative samples
        pos_seqs, neg_seqs = self._create_grouped_sequences(
            max_seq_len=max_seq_len,
            neg_samples=neg_samples,
            seed=seed,
        )

        # Edge case: no valid samples
        if pos_seqs.shape[0] == 0:
            logger.attention(
                "No valid user history samples generated. Ensure users have at least 2 interactions."
            )

        # Cache the generated tensors for future use
        dataset_args = {
            "positive_sequences": pos_seqs,
            "negative_samples": neg_seqs,
        }

        # Create dataset and DataLoader
        dataset = UserHistoryDataset(**dataset_args)
        self._cached_dataset[cache_key] = dataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _create_grouped_sequences(
        self, max_seq_len: int, neg_samples: int, seed: int
    ) -> Tuple[Tensor, Tensor]:
        """Core logic to transform interaction data into user-history sequences
        using a vectorized approach.

        Args:
            max_seq_len (int): Maximum length of the user history sequence.
            neg_samples (int): Number of negative samples for each positive item.
            seed (int): Seed for reproducibility of negative sampling.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Tensor: Padded positive item sequences.
                - Tensor: Padded negative item samples.
        """
        # Get user sessions as lists of item indices
        user_sessions = self._get_user_sessions()
        valid_sessions = user_sessions[user_sessions.str.len() >= 2].apply(
            lambda s: s[-max_seq_len:]
        )

        # Edge case: no valid sessions
        if valid_sessions.empty:
            logger.attention(
                "No valid user history samples generated. Ensure users have at least 2 interactions."
            )
            return torch.empty(0), torch.empty(0)

        # Convert sessions to padded tensors
        pos_seqs_0_indexed_lists = valid_sessions.tolist()
        pos_seqs_tensors = [
            torch.tensor(s, dtype=torch.long) for s in pos_seqs_0_indexed_lists
        ]
        padded_pos_seqs = pad_sequence(
            pos_seqs_tensors, batch_first=True, padding_value=self._niid
        )

        # Retrieve shape information
        num_users, current_max_len = padded_pos_seqs.shape

        # Edge case: if all valid sessions have length < 2
        if current_max_len < 2:
            logger.attention(
                "All valid sessions have length < 2. No negative samples can be generated."
            )
            return padded_pos_seqs, torch.empty(0)

        # Set the seed for negative sampling
        np.random.seed(seed)

        # Generate initial negative candidates
        # Shape: (N, L-1, K)
        neg_candidates_shape = (num_users, current_max_len - 1, neg_samples)
        neg_candidates = np.random.randint(
            0, self._niid, size=neg_candidates_shape, dtype=np.int64
        )
        history_tensor = padded_pos_seqs

        # Prepare masks for collision detection
        j_indices = torch.arange(
            current_max_len - 1, dtype=torch.long
        )  # (L-1): [0, 1, ..., L-2]
        h_indices = torch.arange(
            current_max_len, dtype=torch.long
        )  # (L): [0, 1, ..., L-1]

        # Create masks
        valid_history_mask = (
            h_indices.unsqueeze(0) <= j_indices.unsqueeze(1) + 1
        )  # (L-1, L)
        broadcast_mask = valid_history_mask.unsqueeze(0).unsqueeze(2)  # (1, L-1, 1, L)

        # Calculate sequence lengths and valid target masks
        seq_lengths = (history_tensor != self._niid).sum(dim=1)  # (N,)
        valid_target_mask = j_indices.unsqueeze(0) < (seq_lengths.unsqueeze(1) - 1)
        valid_target_broadcast_mask = valid_target_mask.unsqueeze(-1)  # (N, L-1, K)

        # Check if there are any invalid candidates
        invalid_mask_np = np.ones(neg_candidates_shape, dtype=bool)

        # Iteratively re-sample until no collisions exist
        while np.any(invalid_mask_np):
            neg_candidates_tensor = torch.from_numpy(neg_candidates).long()

            # Collision with positive target
            # N, L-1, K, L)
            collision_matrix = (
                neg_candidates_tensor.unsqueeze(-1)  # (N, L-1, K, 1)
                == history_tensor.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, L)
            )

            # Invalid candidates are those that collide with any item in the history
            relevant_collisions = collision_matrix & broadcast_mask  # (N, L-1, K, L)
            invalid_mask_torch = torch.any(relevant_collisions, dim=3)  # (N, L-1, K)
            invalid_mask_torch = invalid_mask_torch & valid_target_broadcast_mask
            invalid_mask_np = invalid_mask_torch.numpy()

            if not np.any(invalid_mask_np):
                break  # No collisions, continue

            # Re-sample only the invalid candidates
            num_invalid = np.sum(invalid_mask_np)
            new_candidates = np.random.randint(
                0, self._niid, size=num_invalid, dtype=np.int64
            )
            neg_candidates[invalid_mask_np] = new_candidates

        # Convert negative samples to 1-based indexing and apply padding mask
        padded_neg_samples_tensor = torch.from_numpy(
            neg_candidates
        ).long()  # (N, L-1, K)

        # Apply padding mask
        padded_neg_samples_tensor.masked_fill_(~valid_target_broadcast_mask, self._niid)

        return padded_pos_seqs, padded_neg_samples_tensor

    def get_cloze_mask_dataloader(
        self,
        max_seq_len: int,
        mask_prob: float,
        mask_token_id: int,
        neg_samples: int,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        low_memory: bool = False,
    ) -> DataLoader:
        """Creates a DataLoader for cloze mask training task.

        For each user sequence, this method randomly masks some items and prepares
        the original items as positive targets, along with negative samples.

        Args:
            max_seq_len (int): Maximum length of sequences.
            mask_prob (float): Probability of an item being masked.
            mask_token_id (int): The special token ID used for masking.
            neg_samples (int): Number of negative samples per masked item.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.
            seed (int): Seed for reproducibility.
            low_memory (bool): Whether to create the dataloader with a lazy approach.

        Returns:
            DataLoader: A DataLoader yielding (masked_seq, pos_items, neg_items, masked_indices).
        """
        if low_memory:
            user_sessions = self._get_user_sessions()

            lazy_dataset = LazyClozeMaskDataset(
                user_sessions=user_sessions,
                max_seq_len=max_seq_len,
                mask_prob=mask_prob,
                mask_token_id=mask_token_id,
                neg_samples=neg_samples,
                niid=self._niid,
                padding_token_id=self._niid,
                seed=seed,
            )

            if len(lazy_dataset) == 0:
                logger.negative(
                    "LazyClozeMaskDataset is empty. No valid sessions found."
                )
                return DataLoader(torch.utils.data.TensorDataset(torch.empty(0)))

            custom_collate_fn = partial(
                collate_fn_cloze_mask, padding_token_id=self._niid
            )

            return DataLoader(
                lazy_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=custom_collate_fn,
            )

        # Caching for efficiency
        cache_key = f"cloze_len_{max_seq_len}_prob_{mask_prob}_neg_{neg_samples}"
        if cache_key in self._cached_dataset:
            dataset = self._cached_dataset[cache_key]
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        # Create masked sequences and targets
        masked_seq, pos_items, neg_items, masked_indices = (
            self._create_cloze_mask_samples(
                max_seq_len=max_seq_len,
                mask_prob=mask_prob,
                mask_token_id=mask_token_id,
                neg_samples=neg_samples,
                seed=seed,
            )
        )

        # Create Dataset and DataLoader
        dataset = ClozeMaskDataset(masked_seq, pos_items, neg_items, masked_indices)
        self._cached_dataset[cache_key] = dataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _create_cloze_mask_samples(
        self,
        max_seq_len: int,
        mask_prob: float,
        mask_token_id: int,
        neg_samples: int,
        seed: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Core logic for creating masked sequences.

        Args:
            max_seq_len (int): Maximum length of sequences.
            mask_prob (float): Probability of an item being masked.
            mask_token_id (int): The special token ID used for masking.
            neg_samples (int): Number of negative samples per masked item.
            seed (int): Seed for reproducibility.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: A tuple containing:
                - Tensor: Masked item sequences.
                - Tensor: Positive item tensor.
                - Tensor: Negative item tensor.
                - Tensor: Indices of masked items in the sequences.
        """
        np.random.seed(seed)
        user_sessions = self._get_user_sessions()

        # Filter sessions with at less than 2 items
        valid_sessions = (
            user_sessions[user_sessions.str.len() >= 2]
            .apply(lambda s: s[-max_seq_len:])
            .tolist()
        )

        # Edge case: no valid sessions
        if not valid_sessions:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

        # Pad sequences
        seq_tensors = [torch.tensor(s, dtype=torch.long) for s in valid_sessions]
        padded_seqs = pad_sequence(
            seq_tensors, batch_first=True, padding_value=self._niid
        )

        masked_sequences = padded_seqs.clone()
        all_pos_items: List[list] = []
        all_masked_indices: List[list] = []

        max_masked_items = 0

        for i in range(len(padded_seqs)):
            seq = padded_seqs[i]

            # Find valid indices (non-padded)
            valid_indices = (seq != self._niid).nonzero(as_tuple=True)[0]
            if len(valid_indices) == 0:
                all_pos_items.append([])
                all_masked_indices.append([])
                continue

            # Determine number of items to mask
            num_to_mask = max(1, int(np.ceil(len(valid_indices) * mask_prob)))

            # Randomly select indices to mask
            indices_to_mask = np.random.choice(
                valid_indices.numpy(), num_to_mask, replace=False
            )

            # Save positive items and masked indices
            pos_items = seq[indices_to_mask].tolist()
            all_pos_items.append(pos_items)
            all_masked_indices.append(indices_to_mask.tolist())

            # Apply masking
            masked_sequences[i, indices_to_mask] = mask_token_id

            max_masked_items = max(max_masked_items, num_to_mask)

        # Pad positive items and masked indices
        # NOTE: Masked indices will use 0-padding and they will be ignored during loss computation
        padded_pos_items = torch.tensor(
            [p + [self._niid] * (max_masked_items - len(p)) for p in all_pos_items],
            dtype=torch.long,
        )
        padded_masked_indices = torch.tensor(
            [m + [0] * (max_masked_items - len(m)) for m in all_masked_indices],
            dtype=torch.long,
        )

        # Generate negative samples
        num_samples, num_masked = padded_pos_items.shape
        neg_candidates = np.random.randint(
            0, self._niid, size=(num_samples, num_masked, neg_samples), dtype=np.int64
        )

        # Vectorized collision check and re-sampling
        history_sets = [set(s) for s in valid_sessions]
        for i in range(num_samples):
            for j in range(num_masked):
                # If the position is padded, set negatives to padding token
                if padded_pos_items[i, j] == self._niid:
                    neg_candidates[i, j, :] = self._niid
                    continue

                # Check for collisions and re-sample
                for k in range(neg_samples):
                    while neg_candidates[i, j, k] in history_sets[i]:
                        neg_candidates[i, j, k] = np.random.randint(0, self._niid)

        neg_items_tensor = torch.from_numpy(neg_candidates).long()
        return (
            masked_sequences,
            padded_pos_items,
            neg_items_tensor,
            padded_masked_indices,
        )
