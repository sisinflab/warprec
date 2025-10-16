from typing import Optional, Tuple, Dict, List

import torch
import pandas as pd
import numpy as np
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from warprec.utils.logger import logger


class SessionDataset(Dataset):
    """Personalized dataset for session based data.

    Used by sequential models to capture temporal information from
    user interaction history.

    Args:
        sequences (Tensor): The sequence of interactions.
        sequence_lengths (Tensor): The length of the sequence (before padding).
        positive_items (Tensor): The positive items.
        negative_items (Optional[Tensor]): The negative items.
        users (Optional[Tensor]): The user IDs associated with the sequences.
    """

    def __init__(
        self,
        sequences: Tensor,
        sequence_lengths: Tensor,
        positive_items: Tensor,
        negative_items: Optional[Tensor] = None,
        users: Optional[Tensor] = None,
    ):
        self.sequences = sequences
        self.sequence_lengths = sequence_lengths
        self.positive_items = positive_items
        self.negative_items = negative_items
        self.users = users

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.users is not None:
            if self.negative_items is not None:
                return (
                    self.users[idx],
                    self.sequences[idx],
                    self.sequence_lengths[idx],
                    self.positive_items[idx],
                    self.negative_items[idx],
                )
            return (
                self.users[idx],
                self.sequences[idx],
                self.sequence_lengths[idx],
                self.positive_items[idx],
            )
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


class UserHistoryDataset(Dataset):
    """Personalized dataset for user history-based data.

    Used by sequential models that process the entire user interaction history
    as a single sequence.

    Args:
        positive_sequences (Tensor): The sequences of positive interactions.
        negative_samples (Tensor): The sequences of negative interactions.
    """

    def __init__(self, positive_sequences: Tensor, negative_samples: Tensor):
        self.positive_sequences = positive_sequences
        self.negative_samples = negative_samples

    def __len__(self) -> int:
        return len(self.positive_sequences)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.positive_sequences[idx], self.negative_samples[idx]


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
        seed (int): The seed for Numpy number generator used for
            reproducibility of negative sampling.

    Raises:
        ValueError: If the user or item label are not found in the DataFrame.
    """

    _cached_sequential_data: dict = {}
    _cached_grouped_sequential_data: dict = {}
    _cached_user_histories: Dict[int, List[int]] = {}

    def __init__(
        self,
        data: DataFrame,
        user_mapping: dict,
        item_mapping: dict,
        user_id_label: str = "user_id",
        item_id_label: str = "item_id",
        timestamp_label: str = "timestamp",
        seed: int = 42,
    ):
        if user_id_label not in data.columns:
            raise ValueError(f"User column '{user_id_label}' not found in DataFrame.")
        if item_id_label not in data.columns:
            raise ValueError(f"Item column '{item_id_label}' not found in DataFrame.")

        self._inter_df = data
        self._user_label = user_id_label
        self._item_label = item_id_label
        self._timestamp_label = timestamp_label
        self._umap = user_mapping
        self._imap = item_mapping
        self._niid = len(self._imap)

        # Set the random seed
        np.random.seed(seed)

    def clear_history_cache(self):
        """Clears all cached data to free up memory."""
        self._cached_sequential_data = {}
        self._cached_grouped_sequential_data = {}
        self._cached_user_histories = {}

    def _get_user_sessions(self) -> pd.Series:
        """
        Maps, sorts (if timestamp is available), and groups interactions by user.
        This centralized helper method prevents code duplication.
        """
        cols_to_map = {
            self._user_label: self._inter_df[self._user_label].map(self._umap),
            self._item_label: self._inter_df[self._item_label].map(self._imap),
        }
        if self._timestamp_label and self._timestamp_label in self._inter_df.columns:
            cols_to_map[self._timestamp_label] = self._inter_df[self._timestamp_label]

        mapped_df = pd.DataFrame(cols_to_map).dropna()
        mapped_df[[self._user_label, self._item_label]] = mapped_df[
            [self._user_label, self._item_label]
        ].astype(int)

        if self._timestamp_label and self._timestamp_label in self._inter_df.columns:
            mapped_df = mapped_df.sort_values(
                by=[self._user_label, self._timestamp_label]
            )

        return mapped_df.groupby(self._user_label)[self._item_label].agg(list)

    def get_sequential_dataloader(
        self,
        max_seq_len: int,
        num_negatives: int = 0,
        include_user_id: bool = False,
        batch_size: int = 1024,
        shuffle: bool = True,
    ) -> DataLoader:
        """Creates a DataLoader for sequential data."""
        cache_key = (max_seq_len, num_negatives, include_user_id)
        if cache_key in self._cached_sequential_data:
            cached_tensors = self._cached_sequential_data[cache_key]
            dataset = SessionDataset(**cached_tensors)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        (
            tensor_user_id,
            padded_item_seq,
            tensor_item_seq_len,
            tensor_pos_item_id,
            tensor_neg_item_id,
        ) = self._create_sequences_and_targets(
            num_negatives=num_negatives,
            max_seq_len=max_seq_len,
            include_user_id=include_user_id,
        )

        if padded_item_seq.shape[0] == 0:
            logger.attention(
                "No valid sequential samples generated. Ensure sessions have at least 2 interactions."
            )

        dataset_args = {
            "sequences": padded_item_seq,
            "sequence_lengths": tensor_item_seq_len,
            "positive_items": tensor_pos_item_id,
        }
        if tensor_user_id is not None:
            dataset_args["users"] = tensor_user_id
        if tensor_neg_item_id is not None:
            dataset_args["negative_items"] = tensor_neg_item_id
        self._cached_sequential_data[cache_key] = dataset_args

        dataset = SessionDataset(**dataset_args)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _create_sequences_and_targets(
        self,
        num_negatives: int,
        max_seq_len: int,
        include_user_id: bool = False,
    ) -> Tuple[Optional[Tensor], Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Core logic for transforming interaction data into sequential training samples."""
        user_sessions = self._get_user_sessions()
        user_sessions = user_sessions[user_sessions.str.len() >= 2]

        if user_sessions.empty:
            return (
                None if not include_user_id else torch.empty(0, dtype=torch.long),
                torch.empty((0, 0), dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                None if num_negatives == 0 else torch.empty(0, dtype=torch.long),
            )

        # Pre-allocate lists
        all_sequences = []
        all_targets = []
        all_user_ids = []

        # Iterate over the Pandas Series
        for user_id, items in user_sessions.items():
            # Generate all the pairs seq, target for each user
            for i in range(len(items) - 1):
                sequence = items[: i + 1]
                target = items[i + 1]

                # Truncate the sequence at the max_seq_len
                truncated_sequence = sequence[-max_seq_len:]

                # Add the sequence only if not empty
                if truncated_sequence:
                    all_sequences.append(truncated_sequence)
                    all_targets.append(target)
                    if include_user_id:
                        all_user_ids.append(user_id)

        # Create the final dataset
        training_data_dict = {
            "sequences": all_sequences,
            "targets": all_targets,
        }
        if include_user_id:
            training_data_dict[self._user_label] = all_user_ids

        training_data = pd.DataFrame(training_data_dict)

        tensor_user_id = None
        if include_user_id:
            tensor_user_id = torch.tensor(
                training_data[self._user_label].values, dtype=torch.long
            )

        # Handle Negative Sampling if requested
        neg_tensor = None
        if num_negatives > 0:
            num_samples = len(training_data)
            targets_np = training_data["targets"].values
            interacted_sets = training_data["sequences"].apply(set)

            # Sample candidates
            neg_candidates = np.random.randint(
                0, self._niid, size=(num_samples, num_negatives), dtype=np.int64
            )

            # Collision check
            is_target = neg_candidates == targets_np[:, None]

            # Define the 'in_history' mask
            in_history = np.array(
                [
                    [neg in hist for neg in negs]  # type: ignore[attr-defined]
                    for negs, hist in zip(neg_candidates, interacted_sets)
                ]
            )

            invalid_mask = is_target | in_history

            # If invalid indices have been sampled, repeat the process
            while np.any(invalid_mask):
                num_invalid = np.sum(invalid_mask)

                # Sample only non valid candidates
                new_candidates = np.random.randint(
                    0, self._niid, size=num_invalid, dtype=np.int64
                )
                neg_candidates[invalid_mask] = new_candidates

                # Find the indexes that need updating
                rows_to_recheck, _ = np.where(invalid_mask)
                updated_candidates = neg_candidates[invalid_mask]

                # Check is_target only on updated candidates
                is_target_new = updated_candidates == targets_np[rows_to_recheck]

                # Check in_history only on updated candidates
                in_history_new = np.array(
                    [
                        cand in interacted_sets.iloc[row]
                        for cand, row in zip(updated_candidates, rows_to_recheck)
                    ]
                )

                # Combine results
                still_invalid = is_target_new | in_history_new

                # Update invalid mask
                new_invalid_mask = np.zeros_like(invalid_mask)
                new_invalid_mask[invalid_mask] = still_invalid
                invalid_mask = new_invalid_mask

            neg_tensor = torch.tensor(neg_candidates + 1, dtype=torch.long)

        sequences_list_0_indexed = training_data["sequences"].tolist()
        sequences_tensors_1_indexed = [
            torch.tensor(s, dtype=torch.long) + 1 for s in sequences_list_0_indexed
        ]
        padded_item_seq = torch.nn.utils.rnn.pad_sequence(
            sequences_tensors_1_indexed, batch_first=True, padding_value=0
        )
        tensor_item_seq_len = torch.tensor(
            training_data["sequences"].str.len().tolist(), dtype=torch.long
        )
        tensor_pos_item_id = torch.tensor(
            training_data["targets"].values.astype(np.int64) + 1, dtype=torch.long
        )

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
        """Retrieves padded historical sequences for evaluation."""
        if not self._cached_user_histories:
            self._compute_cache_user_history()

        sequences_to_process = []
        lengths_to_process = []
        for uid in user_ids:
            history = self._cached_user_histories.get(uid, [])
            recent_history = history[-max_seq_len:]
            sequences_to_process.append(torch.tensor(recent_history, dtype=torch.long))
            lengths_to_process.append(len(recent_history))

        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            sequences_to_process, batch_first=True, padding_value=0
        )
        sequence_lengths = torch.tensor(lengths_to_process, dtype=torch.long)
        return padded_sequences, sequence_lengths

    def _compute_cache_user_history(self):
        """Computes and caches the complete interaction history for every user."""
        user_sessions = self._get_user_sessions()
        self._cached_user_histories = user_sessions.apply(
            lambda x: (np.array(x) + 1).tolist()
        ).to_dict()

    def get_user_history_dataloader(
        self,
        max_seq_len: int,
        num_negatives: int,
        batch_size: int = 1024,
        shuffle: bool = True,
    ) -> DataLoader:
        """Creates a DataLoader where each item is a user's full history."""
        cache_key = (max_seq_len, num_negatives)
        if cache_key in self._cached_grouped_sequential_data:
            cached_tensors = self._cached_grouped_sequential_data[cache_key]
            dataset = UserHistoryDataset(**cached_tensors)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        pos_seqs, neg_samples = self._create_grouped_sequences(
            max_seq_len=max_seq_len,
            num_negatives=num_negatives,
        )

        if pos_seqs.shape[0] == 0:
            logger.attention(
                "No valid user history samples generated. Ensure users have at least 2 interactions."
            )

        dataset_args = {
            "positive_sequences": pos_seqs,
            "negative_samples": neg_samples,
        }
        self._cached_grouped_sequential_data[cache_key] = dataset_args
        dataset = UserHistoryDataset(**dataset_args)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _create_grouped_sequences(
        self, max_seq_len: int, num_negatives: int
    ) -> Tuple[Tensor, Tensor]:
        """Core logic to transform interaction data into user-history sequences."""
        user_sessions = self._get_user_sessions()

        processed_sessions = (
            user_sessions[user_sessions.str.len() >= 2]
            .apply(lambda s: s[-max_seq_len:])
            .tolist()
        )

        if not processed_sessions:
            return torch.empty(0), torch.empty(0)

        all_negative_samples = []
        for session in processed_sessions:
            seq_len = len(session)
            session_negatives = np.zeros((seq_len - 1, num_negatives), dtype=np.int64)
            for i in range(1, seq_len):
                history = set(session[: i + 1])
                step_negatives: list = []
                while len(step_negatives) < num_negatives:
                    candidates = np.random.randint(
                        0, self._niid, size=num_negatives * 2
                    )
                    valid_candidates = [c for c in candidates if c not in history]
                    step_negatives.extend(valid_candidates)
                session_negatives[i - 1] = step_negatives[:num_negatives]
            all_negative_samples.append(torch.from_numpy(session_negatives))

        pos_sequences_1_indexed = [
            torch.tensor(s, dtype=torch.long) + 1 for s in processed_sessions
        ]
        padded_pos_sequences = torch.nn.utils.rnn.pad_sequence(
            pos_sequences_1_indexed, batch_first=True, padding_value=0
        )
        neg_samples_1_indexed = [neg + 1 for neg in all_negative_samples]
        padded_neg_samples = torch.nn.utils.rnn.pad_sequence(
            neg_samples_1_indexed, batch_first=True, padding_value=0
        )

        return padded_pos_sequences, padded_neg_samples
