from typing import Optional, Tuple, Dict, List, Any

import torch
import pandas as pd
import numpy as np
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from warprec.utils.logger import logger


class SessionDataset(Dataset):
    """Personalized dataset for session-based data.

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
    as a single sequence, often for BPR-style loss.

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
    ):
        # Validate that provided columns exist in the DataFrame
        if user_id_label not in data.columns:
            raise ValueError(f"User column '{user_id_label}' not found in DataFrame.")
        if item_id_label not in data.columns:
            raise ValueError(f"Item column '{item_id_label}' not found in DataFrame.")

        self._inter_df = data

        # Data labels
        self._user_label = user_id_label
        self._item_label = item_id_label
        self._timestamp_label = timestamp_label

        # User and item mappings
        self._umap = user_mapping
        self._imap = item_mapping

        # The number of items is derived from the mapping
        self._niid = len(self._imap)

    def clear_history_cache(self):
        """Clears all cached data to free up memory."""
        self._cached_sequential_data = {}
        self._cached_grouped_sequential_data = {}
        self._cached_user_histories = {}

    def _get_user_sessions(self) -> pd.Series:
        """
        Maps, sorts (if timestamp is available), and groups interactions by user.
        This centralized helper method prevents code duplication.

        Returns:
            pd.Series: A Series where index is the mapped user_id and values are
                       lists of mapped item_ids.
        """
        cols_to_map = {
            self._user_label: self._inter_df[self._user_label].map(self._umap),
            self._item_label: self._inter_df[self._item_label].map(self._imap),
        }
        if self._timestamp_label in self._inter_df.columns:
            cols_to_map[self._timestamp_label] = self._inter_df[self._timestamp_label]

        mapped_df = pd.DataFrame(cols_to_map).dropna()
        mapped_df[[self._user_label, self._item_label]] = mapped_df[
            [self._user_label, self._item_label]
        ].astype(int)

        if self._timestamp_label in self._inter_df.columns:
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
        """Creates a DataLoader for sequential data.

        Each sample consists of a historical sequence and the next item as the target.

        Args:
            max_seq_len (int): Maximum length of sequences.
            num_negatives (int): Number of negative samples per positive target.
            include_user_id (bool): If True, the user ID is included in each batch.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: A DataLoader that yields batches of tensors.
                        Format depends on `num_negatives` and `include_user_id`.
        """
        cache_key = (max_seq_len, num_negatives, include_user_id)
        if cache_key in self._cached_sequential_data:
            cached_tensors = self._cached_sequential_data[cache_key]
            dataset = SessionDataset(**cached_tensors)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        tensors = self._create_sequences_and_targets(
            num_negatives=num_negatives,
            max_seq_len=max_seq_len,
            include_user_id=include_user_id,
        )
        (
            tensor_user_id,
            padded_item_seq,
            tensor_item_seq_len,
            tensor_pos_item_id,
            tensor_neg_item_id,
        ) = tensors

        if padded_item_seq.shape[0] == 0:
            logger.attention(
                "No valid sequential samples generated. Ensure sessions have at least 2 interactions."
            )

        dataset_args = {
            "sequences": padded_item_seq,
            "sequence_lengths": tensor_item_seq_len,
            "positive_items": tensor_pos_item_id,
        }
        if include_user_id:
            dataset_args["users"] = tensor_user_id
        if num_negatives > 0:
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
        """Core logic to generate sequences and targets from user interactions."""
        user_sessions = self._get_user_sessions()
        user_sessions = user_sessions[user_sessions.str.len() >= 2]

        if user_sessions.empty:
            return (
                torch.empty(0, dtype=torch.long) if include_user_id else None,
                torch.empty((0, max_seq_len), dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                torch.empty((0, num_negatives), dtype=torch.long)
                if num_negatives > 0
                else None,
            )

        # For a session [i1, i2, i3, i4], this generates pairs:
        # - seq: [i1], target: i2
        # - seq: [i1, i2], target: i3
        # - seq: [i1, i2, i3], target: i4
        sequences = user_sessions.apply(lambda s: [s[:i] for i in range(1, len(s))])
        targets = user_sessions.apply(lambda s: s[1:])

        session_df = pd.DataFrame({"sequences": sequences, "targets": targets})
        training_data = session_df.explode(["sequences", "targets"]).reset_index()

        # Truncate sequences to max_seq_len
        training_data["sequences"] = training_data["sequences"].apply(
            lambda s: s[-max_seq_len:]
        )

        tensor_user_id = (
            torch.tensor(training_data[self._user_label].values, dtype=torch.long)
            if include_user_id
            else None
        )

        # Handle Negative Sampling
        neg_tensor = None
        if num_negatives > 0:
            num_samples = len(training_data)
            targets_np = training_data["targets"].values
            interacted_sets = training_data["sequences"].apply(set)

            neg_candidates = np.random.randint(
                0, self._niid, size=(num_samples, num_negatives), dtype=np.int64
            )

            # A negative sample is invalid if it's the positive target
            is_target = neg_candidates == targets_np[:, None]

            # Or if it's already in the user's history for that sequence
            in_history = np.array(
                [
                    [neg in hist for neg in negs]  # type: ignore[attr-defined]
                    for negs, hist in zip(neg_candidates, interacted_sets)
                ]
            )

            invalid_mask = is_target | in_history

            while np.any(invalid_mask):
                num_invalid = np.sum(invalid_mask)
                # Resample only for the invalid entries
                new_candidates = np.random.randint(
                    0, self._niid, size=num_invalid, dtype=np.int64
                )
                neg_candidates[invalid_mask] = new_candidates

                # Re-check the validity ONLY for the positions that were just updated
                rows_to_recheck, _ = np.where(invalid_mask)
                updated_candidates = neg_candidates[invalid_mask]

                is_target_new = updated_candidates == targets_np[rows_to_recheck]
                in_history_new = np.array(
                    [
                        cand in interacted_sets.iloc[row]
                        for cand, row in zip(updated_candidates, rows_to_recheck)
                    ]
                )

                still_invalid = is_target_new | in_history_new

                # Create a new mask of the original shape and update it
                new_invalid_mask = np.zeros_like(invalid_mask)
                new_invalid_mask[invalid_mask] = still_invalid
                invalid_mask = new_invalid_mask

            # Item indices are 0-indexed; models often expect 1-indexed items with 0 for padding
            neg_tensor = torch.tensor(neg_candidates + 1, dtype=torch.long)

        # Convert sequences to 1-indexed tensors for the model
        sequences_list = training_data["sequences"].tolist()
        sequences_tensors = [
            torch.tensor(s, dtype=torch.long) + 1 for s in sequences_list
        ]

        padded_item_seq = pad_sequence(
            sequences_tensors, batch_first=True, padding_value=0
        )
        tensor_item_seq_len = torch.tensor(
            [len(s) for s in sequences_list], dtype=torch.long
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

    def _compute_cache_user_history(self):
        """Computes and caches the complete interaction history for every user."""
        user_sessions = self._get_user_sessions()
        self._cached_user_histories = user_sessions.apply(
            lambda x: (np.array(x) + 1).tolist()
        ).to_dict()

    def get_user_history_sequences(
        self, user_ids: List[int], max_seq_len: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Retrieves padded historical sequences for a list of user IDs for evaluation.
        Sequences are 1-indexed, padded with 0.

        Args:
            user_ids (List[int]): A list of user indices.
            max_seq_len (int): Maximum length of sequences.

        Returns:
            Tuple[Tensor, Tensor]: Padded item sequences and their original lengths.
        """
        if not self._cached_user_histories:
            self._compute_cache_user_history()

        sequences, lengths = [], []
        for uid in user_ids:
            history = self._cached_user_histories.get(uid, [])
            truncated_history = history[-max_seq_len:]
            sequences.append(torch.tensor(truncated_history, dtype=torch.long))
            lengths.append(len(truncated_history))

        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
        sequence_lengths = torch.tensor(lengths, dtype=torch.long)

        return padded_sequences, sequence_lengths

    def get_user_history_dataloader(
        self,
        max_seq_len: int,
        num_negatives: int,
        batch_size: int = 1024,
        shuffle: bool = True,
    ) -> DataLoader:
        """Creates a DataLoader where each item is a user's full history.

        Args:
            max_seq_len (int): Maximum length of sequences.
            num_negatives (int): Number of negative samples per positive item in the history.
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: Yields batches of (positive_sequences, negative_samples).
        """
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

        dataset_args = {"positive_sequences": pos_seqs, "negative_samples": neg_samples}
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
            session_negatives = []
            for i in range(1, len(session)):
                # History includes the current target item
                history_set = set(session[: i + 1])
                step_negatives: List[Any] = []
                while len(step_negatives) < num_negatives:
                    # Generate more candidates than needed to increase chance of finding valid ones
                    candidates = np.random.randint(
                        0, self._niid, size=num_negatives * 5
                    )
                    # Use fast, vectorized np.isin for filtering
                    valid_mask = ~np.isin(candidates, list(history_set))
                    step_negatives.extend(candidates[valid_mask])

                session_negatives.append(step_negatives[:num_negatives])

            all_negative_samples.append(
                torch.tensor(session_negatives, dtype=torch.long)
            )

        # Convert to 1-indexed sequences and pad
        pos_sequences_1i = [
            torch.tensor(s, dtype=torch.long) + 1 for s in processed_sessions
        ]
        padded_pos_sequences = pad_sequence(
            pos_sequences_1i, batch_first=True, padding_value=0
        )

        neg_samples_1i = [neg + 1 for neg in all_negative_samples]
        padded_neg_samples = pad_sequence(
            neg_samples_1i, batch_first=True, padding_value=0
        )

        return padded_pos_sequences, padded_neg_samples
