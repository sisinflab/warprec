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


class GroupSessionDataset(Dataset):
    """Personalized dataset for grouped session based data.

    Used by sequential models to capture temporal information from
    entire user interaction history.

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
    """Sessions class will handle the data of the sessions for
    session-based recommendations.

    Args:
        data (DataFrame): Transaction data in DataFrame format.
        user_mapping (dict): Mapping of user ID -> user idx.
        item_mapping (dict): Mapping of item ID -> item idx.
        batch_size (int): The batch size that will be used to
            iterate over the interactions.
        timestamp_label (str): The label of the timestamp column.
    """

    _cached_sequential_data: dict = {}
    _cached_user_histories: Dict[int, List[int]] = {}

    def __init__(
        self,
        data: DataFrame,
        user_mapping: dict,
        item_mapping: dict,
        batch_size: int = 1024,
        timestamp_label: str = None,
    ):
        # Setup the variables
        self._inter_df = data
        self.batch_size = batch_size

        # Set DataFrame labels
        self._user_label = data.columns[0]
        self._item_label = data.columns[1]
        self._timestamp_label = timestamp_label

        # Definition of dimensions
        self._niid = self._inter_df[self._item_label].nunique()

        # Set mappings
        self._umap = user_mapping
        self._imap = item_mapping

    def clear_history_cache(self):
        """This method will can be used to clear the
        cached sequential data if not used anymore.
        """
        self._cached_sequential_data = {}
        self._cached_user_histories = {}

    def get_sequential_dataloader(
        self,
        max_seq_len: int,
        num_negatives: int = 0,
        user_id: bool = False,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a dataloader for sequential data.

        Args:
            max_seq_len (int): Maximum length of sequences produced.
            num_negatives (int): Number of negative samples per user.
            user_id (bool): Wether or not to return also the user_id.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: Yields (item_seq, item_seq_len, pos_item_id) if num_negatives = 0.
                Yields (item_seq, item_seq_len, pos_item_id, neg_item_id) if num_negatives > 0.
        """
        # Check if sequential data has already been computed
        # and is stored in cache
        cache_key = (num_negatives, user_id)
        if cache_key in self._cached_sequential_data:
            cached_tensors = self._cached_sequential_data[cache_key]
            dataset = SessionDataset(**cached_tensors)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        # Call the optimized processing function
        (
            tensor_user_id,
            padded_item_seq,
            tensor_item_seq_len,
            tensor_pos_item_id,
            tensor_neg_item_id,
        ) = self._create_sequences_and_targets(
            num_negatives=num_negatives,
            max_seq_len=max_seq_len,
            user_id=user_id,
        )

        # Check for empty results
        if padded_item_seq.shape[0] == 0:
            logger.attention(
                "No valid sequential samples generated for DataLoader. "
                "Check your data or session definition (min length 2)."
            )

        # Cache the results inside a dictionary
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

        # Create instance of the dataset
        dataset = SessionDataset(**dataset_args)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _create_sequences_and_targets(
        self,
        num_negatives: int,
        max_seq_len: int,
        user_id: bool = False,
    ) -> Tuple[Optional[Tensor], Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Core logic for transforming interaction data into sequential training samples.

        This function uses pandas and numpy for efficient processing.

        Args:
            num_negatives (int): Number of negative samples per user.
            max_seq_len (int): Maximum length of sequences produced.
            user_id (bool): Wether or not to return also the user_id.

        Returns:
            Tuple[Optional[Tensor], Tensor, Tensor, Tensor, Optional[Tensor]]: A tuple containing:
                - Optional[Tensor]: User ID.
                - Tensor: Padded item sequence.
                - Tensor: Item sequence length.
                - Tensor: Positive item tensor.
                - Optional[Tensor]: Negative item tensor.
        """
        mapped_df = pd.DataFrame(
            {
                self._user_label: self._inter_df[self._user_label].map(self._umap),
                self._item_label: self._inter_df[self._item_label].map(self._imap),
                self._timestamp_label: self._inter_df[self._timestamp_label],
            }
        ).dropna()  # Drop any interactions if user/item not in map

        # Convert to integer types
        mapped_df[[self._user_label, self._item_label]] = mapped_df[
            [self._user_label, self._item_label]
        ].astype(int)

        # Group interactions based on timestamp
        user_sessions = (
            mapped_df.sort_values(by=[self._user_label, self._timestamp_label])
            .groupby(self._user_label)[self._item_label]
            .agg(list)
        )

        # Filter out sessions with less than 2 interactions
        user_sessions = user_sessions[user_sessions.str.len() >= 2]

        # Edge case: No user has at least 2 interactions in sequence
        if user_sessions.empty:
            return (
                None if not user_id else torch.empty(0, dtype=torch.long),
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

        def truncate_user_sequences(sequences_for_user, targets_for_user, max_len):
            """Helper function to truncate sequences to max_len."""
            truncated_sequences = []
            truncated_targets = []
            for seq, target in zip(sequences_for_user, targets_for_user):
                # Truncate sequences taking only the last max_len elements
                current_truncated_seq = seq[-max_len:] if max_len > 0 else []
                if current_truncated_seq:
                    truncated_sequences.append(current_truncated_seq)
                    truncated_targets.append(target)
            return truncated_sequences, truncated_targets

        # Truncate each row
        session_df[["sequences", "targets"]] = session_df.apply(
            lambda row: truncate_user_sequences(
                row["sequences"], row["targets"], max_seq_len
            ),
            axis=1,
            result_type="expand",
        )

        # Remove empty rows
        session_df = session_df[session_df["sequences"].apply(lambda x: len(x) > 0)]

        # Explode the lists into separate rows
        training_data = session_df.explode(["sequences", "targets"]).reset_index()

        # Optionally associate every sequence to a user_id
        tensor_user_id = None
        if user_id:
            # The user_id is inside the correct column thanks to the reset_index
            tensor_user_id = torch.tensor(
                training_data[self._user_label].values, dtype=torch.long
            )

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
        """Retrieves padded historical sequences and their lengths for a given list of user IDs.
        Sequences are 1-indexed.

        Args:
            user_ids (List[int]): A list of global user indices.
            max_seq_len (int): Maximum length of sequences produced.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Padded item sequences [num_users, max_seq_len]
                - Sequence lengths [num_users]
        """
        if not self._cached_user_histories:
            self._compute_cache_user_history()

        sequences_to_process = []
        lengths_to_process = []

        for uid in user_ids:
            history = self._cached_user_histories.get(
                uid, []
            )  # Get history, empty if no interactions
            recent_history = history[-max_seq_len:]  # Take only the most recent

            # For prediction, we use the entire history available in the training set
            # The length check can be 0 here as padding handles empty sequences.
            sequences_to_process.append(torch.tensor(recent_history, dtype=torch.long))
            lengths_to_process.append(len(recent_history))

        # Handle cases where some users in the batch might not have history (empty lists)
        # Pad sequences (0-indexed padding value)
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            sequences_to_process, batch_first=True, padding_value=0
        )
        sequence_lengths = torch.tensor(lengths_to_process, dtype=torch.long)

        return padded_sequences, sequence_lengths

    def _compute_cache_user_history(self):
        # First call or cache cleared: pre-process all user histories
        mapped_df_for_history = pd.DataFrame(
            {
                self._user_label: self._inter_df[self._user_label].map(self._umap),
                self._item_label: self._inter_df[self._item_label].map(self._imap),
                self._timestamp_label: self._inter_df[self._timestamp_label],
            }
        ).dropna()  # Ensure user/item are mapped

        mapped_df_for_history[[self._user_label, self._item_label]] = (
            mapped_df_for_history[[self._user_label, self._item_label]].astype(int)
        )

        mapped_df_for_history = mapped_df_for_history.sort_values(
            by=[self._user_label, self._timestamp_label]
        )

        # Store 1-indexed sequences
        self._cached_user_histories = (
            mapped_df_for_history.groupby(self._user_label)[self._item_label]
            .apply(
                lambda x: (x.values + 1).tolist()
            )  # Convert to list and 1-index items
            .to_dict()
        )

    def get_group_sequential_dataloader(
        self,
        max_seq_len: int,
        num_negatives: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Creates a sequential DataLoader grouped by user.

        Args:
            max_seq_len (int): Maximum length of sequences produced.
            num_negatives (int): Number of negative samples per user.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: Yields (pos_item_id, neg_item_id).
        """
        pos_seqs, neg_samples = self._create_grouped_sequences(
            max_seq_len=max_seq_len,
            num_negatives=num_negatives,
        )

        if pos_seqs.shape[0] == 0:
            logger.attention(
                "No valid sequential samples generated for DataLoader. "
                "Check your data or session definition (min length 2)."
            )

        dataset = GroupSessionDataset(pos_seqs, neg_samples)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _create_grouped_sequences(
        self, max_seq_len: int, num_negatives: int
    ) -> Tuple[Tensor, Tensor]:
        """Core logic to transform interaction data into
        grouped sequences.

        Args:
            max_seq_len (int): Maximum length of sequences produced.
            num_negatives (int): Number of negative samples per user.

        Returns:
            Tuple[Tensor, Tensor]:
                - Tensor: Positive sample.
                - Tensor: Negative sample.
        """
        # Retrieve mapped and ordered DataFrame
        mapped_df = pd.DataFrame(
            {
                self._user_label: self._inter_df[self._user_label].map(self._umap),
                self._item_label: self._inter_df[self._item_label].map(self._imap),
                self._timestamp_label: self._inter_df[self._timestamp_label],
            }
        ).dropna()
        mapped_df[[self._user_label, self._item_label]] = mapped_df[
            [self._user_label, self._item_label]
        ].astype(int)

        # Group-by user session
        user_sessions = (
            mapped_df.sort_values(by=[self._user_label, self._timestamp_label])
            .groupby(self._user_label)[self._item_label]
            .apply(list)
        )

        # Filter out user with less than 2 interactions (only one interaction
        # can't be considered a sequence). Truncate sequences to max_seq_len
        processed_sessions = (
            user_sessions[user_sessions.str.len() >= 2]
            .apply(lambda s: s[-max_seq_len:])
            .tolist()
        )

        if not processed_sessions:
            return torch.empty(0), torch.empty(0)

        # Compute negative sampling
        all_negative_samples = []
        for session in processed_sessions:
            seq_len = len(session)
            session_negatives = np.zeros(
                (seq_len - 1, num_negatives), dtype=np.int64
            )  # [L-1, num_negatives]

            for i in range(1, seq_len):
                history = set(
                    session[: i + 1]
                )  # Other items in history cannot be a negative_sample

                # Randomly sample until found
                step_negatives: list = []
                while len(step_negatives) < num_negatives:
                    candidates = np.random.randint(
                        0, self._niid, size=num_negatives * 2
                    )
                    valid_candidates = [c for c in candidates if c not in history]
                    step_negatives.extend(valid_candidates)

                session_negatives[i - 1] = step_negatives[:num_negatives]

            all_negative_samples.append(torch.from_numpy(session_negatives))

        # Shift sequences by 1 (0 is for padding) and add padding
        pos_sequences_1_indexed = [
            torch.tensor(s, dtype=torch.long) + 1 for s in processed_sessions
        ]
        padded_pos_sequences = torch.nn.utils.rnn.pad_sequence(
            pos_sequences_1_indexed, batch_first=True, padding_value=0
        )

        # The same applies to the negatives
        neg_samples_1_indexed = [neg + 1 for neg in all_negative_samples]
        padded_neg_samples = torch.nn.utils.rnn.pad_sequence(
            neg_samples_1_indexed, batch_first=True, padding_value=0
        )

        return padded_pos_sequences, padded_neg_samples
