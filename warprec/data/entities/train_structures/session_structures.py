from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pandas import DataFrame, Series


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


class LazySessionDataset(Dataset):
    """A PyTorch Dataset for sequential recommendation data that generates samples on-the-fly.

    This dataset is designed for low-memory environments. Instead of pre-computing and
    storing all possible training samples (sequences, targets, negative items) in memory,
    it generates each sample individually when its `__getitem__` method is called.

    This approach significantly reduces RAM usage at the cost of higher CPU overhead during
    training batch creation.

    Args:
        sorted_df (DataFrame): The input DataFrame containing interaction data.
            It must be sorted by user ID and then by timestamp.
        user_label (str): The column name for user IDs in `sorted_df`.
        item_label (str): The column name for item IDs in `sorted_df`.
        max_seq_len (int): The maximum length of the input sequence for the model.
        neg_samples (int): The number of negative items to sample for each positive item.
        niid (int): The total number of unique items in the dataset. This is required
            for the upper bound of the random negative sampler.
        include_user_id (bool): If True, the user ID will be included in each sample.
        seed (int): A random seed to ensure reproducibility of negative sampling.
    """

    def __init__(
        self,
        sorted_df: DataFrame,
        user_label: str,
        item_label: str,
        max_seq_len: int,
        neg_samples: int,
        niid: int,
        include_user_id: bool = False,
        seed: int = 42,
    ):
        # Store configuration parameters
        self.max_seq_len = max_seq_len
        self.neg_samples = neg_samples
        self.niid = niid  # Number of unique items
        self.include_user_id = include_user_id
        self.seed = seed

        # Working with NumPy arrays is significantly faster than pandas Series
        self.all_users = sorted_df[user_label].values
        self.all_items_0_indexed = sorted_df[item_label].values

        # A new session begins where the user ID changes
        is_new_session = np.diff(self.all_users, prepend=-1) != 0
        session_starts_idx = np.where(is_new_session)[0]
        session_lengths = np.diff(session_starts_idx, append=len(self.all_users))

        # A session must have at least 2 items to form one (sequence, target) pair
        valid_session_mask = session_lengths >= 2
        self.valid_session_starts = session_starts_idx[valid_session_mask]
        valid_session_lengths = session_lengths[valid_session_mask]

        # --- Create a lightweight index map ---

        # For a session of length L, we can generate L-1 training samples
        num_samples_per_session = valid_session_lengths - 1
        self.total_samples = np.sum(num_samples_per_session)

        if self.total_samples > 0:
            # `sample_to_session_idx[i]` gives the index (in `self.valid_session_starts`)
            # of the session corresponding to the i-th training sample
            self.sample_to_session_idx = np.repeat(
                np.arange(len(self.valid_session_starts)), num_samples_per_session
            )

            # `sequence_index_offsets[i]` gives the position of the target item
            # relative to the start of its session
            self.sequence_index_offsets = np.concatenate(
                [np.arange(n) for n in num_samples_per_session]
            )
        else:
            # Handle edge case where no valid sessions exist.
            self.sample_to_session_idx = np.array([], dtype=np.int64)
            self.sequence_index_offsets = np.array([], dtype=np.int64)

    def __len__(self) -> int:
        """Returns the total number of training samples that can be generated."""
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        """Generates and returns a single training sample on-the-fly.

        Args:
            idx (int): The index of the sample to generate.

        Returns:
            Tuple[Tensor, ...]: A tuple containing the training data. The exact contents depend on the
            `include_user_id` and `num_negatives` flags, but the order is:
            (user_id?, sequence, sequence_length, positive_item, negative_items?)
        """
        # Find the session and target item for the requested sample index
        session_idx = self.sample_to_session_idx[idx]
        sequence_offset = self.sequence_index_offsets[idx]

        session_start_abs = self.valid_session_starts[session_idx]
        target_idx_abs = session_start_abs + sequence_offset + 1

        # Extract the input sequence and the positive target item.
        # The sequence start is truncated by max_seq_len if the history is too long
        seq_start_abs = max(session_start_abs, target_idx_abs - self.max_seq_len)

        # The input sequence is everything up to the target item (0-indexed)
        input_seq_0_indexed = self.all_items_0_indexed[seq_start_abs:target_idx_abs]
        seq_len = len(input_seq_0_indexed)

        # The positive target is the next item in the sequence (0-indexed)
        pos_item_0_indexed = self.all_items_0_indexed[target_idx_abs]

        # Pad the sequence to `max_seq_len`
        padded_seq = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_seq[:seq_len] = input_seq_0_indexed

        # Convert to 1-based indexing for the model. Padding value `0` is reserved
        padded_seq_1_based = torch.from_numpy(padded_seq + 1).long()

        # Explicitly set the padded part to 0
        padded_seq_1_based[seq_len:] = 0

        # Perform negative sampling
        neg_items_1_based = None
        if self.neg_samples > 0:
            # Use a unique, deterministic random number generator for each sample.
            # Seeding with a tuple `(global_seed, sample_index)` ensures that
            # `__getitem__(i)` will always produce the same negative samples
            rng = np.random.default_rng(seed=(self.seed, idx))

            # The forbidden set includes the positive target and all items in its history
            history_set = set(input_seq_0_indexed)
            history_set.add(pos_item_0_indexed)

            neg_candidates = np.zeros(self.neg_samples, dtype=np.int64)

            # Keep sampling until a valid negative item (not in history) is found
            for i in range(self.neg_samples):
                candidate = rng.integers(0, self.niid)
                while candidate in history_set:
                    candidate = rng.integers(0, self.niid)
                neg_candidates[i] = candidate

            # Convert final negative samples to 1-based indexing
            neg_items_1_based = torch.from_numpy(neg_candidates + 1).long()

        # Assemble the final output tuple
        pos_item_1_based = torch.tensor(pos_item_0_indexed + 1, dtype=torch.long)
        tensor_seq_len = torch.tensor(seq_len, dtype=torch.long)

        # Build the output tuple based on the configuration
        output = []
        if self.include_user_id:
            user_id = self.all_users[target_idx_abs]
            output.append(torch.tensor(user_id, dtype=torch.long))

        output.extend([padded_seq_1_based, tensor_seq_len, pos_item_1_based])

        if neg_items_1_based is not None:
            output.append(neg_items_1_based)

        return tuple(output)


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


class LazyUserHistoryDataset(Dataset):
    """A PyTorch Dataset for user history data that generates samples on-the-fly.

    This dataset is the low-memory alternative for UserHistoryDataset. Instead of
    pre-computing and storing all positive sequences and their corresponding negative
    samples, it generates them individually when `__getitem__` is called.

    This is particularly useful for models that require negative samples for every
    timestamp in a user's history, which can lead to very large tensors.

    Args:
        user_sessions (Series): A Series where the index is the user ID and the
            value is a list of their 0-indexed item interactions.
        max_seq_len (int): The maximum length of the user history sequence.
        neg_samples (int): The number of negative items to sample for each positive item
            in the sequence.
        niid (int): The total number of unique items for negative sampling.
        seed (int): A random seed to ensure reproducibility.
    """

    def __init__(
        self,
        user_sessions: Series,
        max_seq_len: int,
        neg_samples: int,
        niid: int,
        seed: int = 42,
    ):
        self.max_seq_len = max_seq_len
        self.neg_samples = neg_samples
        self.niid = niid
        self.seed = seed

        # Filter for sessions long enough to have at least one target item (len >= 2)
        # and store them as a list for efficient indexing
        self.sessions = user_sessions[user_sessions.str.len() >= 2].tolist()

    def __len__(self) -> int:
        """Returns the total number of users with valid histories."""
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Generates and returns a single user's history and negative samples.

        Args:
            idx (int): The index of the user session to process.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing
            - Tensor: The padded positive item sequence (1-based).
            - Tensor: The padded negative item samples (1-based).
        """
        # Get the user's full session and truncate it
        session_0_indexed = self.sessions[idx][-self.max_seq_len :]
        current_len = len(session_0_indexed)

        # Generate negative samples for each step in the sequence
        # The shape will be (current_len - 1, neg_samples)
        neg_candidates = np.zeros((current_len - 1, self.neg_samples), dtype=np.int64)

        # Iterate from the second item onwards, as the first item has no history
        for j in range(1, current_len):
            # Create a unique, deterministic RNG for this specific user and timestamp
            # to ensure reproducibility
            rng = np.random.default_rng(seed=(self.seed, idx, j))

            # The forbidden set includes all items in the history up to the current positive item
            forbidden_set = set(session_0_indexed[: j + 1])

            for k in range(self.neg_samples):
                candidate = rng.integers(0, self.niid)
                while candidate in forbidden_set:
                    candidate = rng.integers(0, self.niid)
                neg_candidates[j - 1, k] = candidate

        # Pad positive sequence
        padded_pos_seq = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_pos_seq[:current_len] = session_0_indexed

        # Pad negative sequences
        # The target shape is (max_seq_len - 1, neg_samples)
        padded_neg_seqs = np.zeros(
            (self.max_seq_len - 1, self.neg_samples), dtype=np.int64
        )
        if current_len > 1:
            padded_neg_seqs[: current_len - 1, :] = neg_candidates

        # Convert to 1-based PyTorch tensors
        # Positive sequence: add 1, but keep padding as 0
        pos_tensor = torch.from_numpy(padded_pos_seq + 1).long()
        pos_tensor[current_len:] = 0

        # Negative sequences: add 1, padding remains 0
        neg_tensor = torch.from_numpy(padded_neg_seqs + 1).long()
        if current_len > 1:
            neg_tensor[current_len - 1 :, :] = 0

        return pos_tensor, neg_tensor
