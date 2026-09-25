from typing import Optional, List, Any, Tuple, Dict

import torch
import numpy as np
import narwhals as nw
from narwhals.dataframe import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix

from warprec.data.entities.common import (
    ITEM_INDEX,
    USER_INDEX,
    map_to_index_space,
    seeded_dataloader,
)
from warprec.data.entities.train_structures import (
    SequentialDataset,
    SameTargetSequentialDataset,
    SlidingWindowDataset,
    ClozeDataset,
)


class Sessions:
    # pylint: disable = too-many-instance-attributes  # this class is the state it holds
    """
    Handles session-based data preparation for sequential recommenders.
    Transforms user-item interactions into padded sequences or sliding windows.
    """

    def __init__(
        self,
        data: DataFrame[Any],
        user_mapping: dict,
        item_mapping: dict,
        sparse_matrix: csr_matrix,
        user_id_label: str = "user_id",
        item_id_label: str = "item_id",
        timestamp_label: str = "timestamp",
        context_labels: Optional[List[str]] = None,
    ):
        # Validation
        if user_id_label not in data.columns:
            raise ValueError(f"User column '{user_id_label}' not found.")
        if item_id_label not in data.columns:
            raise ValueError(f"Item column '{item_id_label}' not found.")

        # Configuration
        self._inter_df = data
        self._umap = user_mapping
        self._imap = item_mapping
        self.user_label = user_id_label
        self.item_label = item_id_label
        self.timestamp_label = timestamp_label
        self.context_labels = context_labels or []

        # Dimensions & Cache
        self._niid = len(self._imap)
        self._nuid = len(self._umap)
        self._cached_user_histories: Dict[int, List[int]] = {}
        self._processed_df: DataFrame[Any] = None  # Cache for the sorted dataframe

        # Internal Structures (Lazy Loaded)
        self._flat_items: Optional[np.ndarray] = None
        self._flat_users: Optional[np.ndarray] = None
        self._user_offsets: Optional[np.ndarray] = None
        self._valid_sample_indices: Optional[np.ndarray] = None
        self._inter_sparse = sparse_matrix

        # Build Core Structures
        self._build_flat_structures()

    def _get_processed_data(self) -> DataFrame[Any]:
        """
        Centralized pipeline: Map IDs -> Drop Missing -> Sort by User/Time.
        Returns a cached Narwhals DataFrame.
        """
        if self._processed_df is not None:
            return self._processed_df

        # Join and Map
        mapped_df = map_to_index_space(
            self._inter_df,
            self.user_label,
            self.item_label,
            self._umap,
            self._imap,
        ).select(
            [
                nw.col(USER_INDEX).alias(self.user_label).cast(nw.Int64),
                nw.col(ITEM_INDEX).alias(self.item_label).cast(nw.Int64),
                # Keep timestamp if exists
                *(
                    [nw.col(self.timestamp_label)]
                    if self.timestamp_label in self._inter_df.columns
                    else []
                ),
                # Keep context if exists. The columns are carried through as
                # they are: a numeric field would lose its value to an integer
                # cast, and a multi-valued one holds several indices per cell.
                *(
                    [
                        nw.col(c)
                        for c in self.context_labels
                        if c in self._inter_df.columns
                    ]
                ),
            ]
        )

        # Sort
        sort_cols = [self.user_label]
        if self.timestamp_label in self._inter_df.columns:
            sort_cols.append(self.timestamp_label)

        self._processed_df = mapped_df.sort(sort_cols)
        return self._processed_df

    def _build_flat_structures(self):
        """
        Converts the processed DataFrame into flat Numpy arrays ("The Tape")
        and calculates user offsets for O(1) access to any user's history.
        """
        df = self._get_processed_data()

        # Extract columns to numpy (The Tape)
        self._flat_users = df.select(self.user_label).to_numpy().flatten()
        self._flat_items = df.select(self.item_label).to_numpy().flatten()

        # Calculate Offsets
        # unique_users are sorted because df is sorted by user
        unique_users, start_indices = np.unique(self._flat_users, return_index=True)

        self._user_offsets = np.zeros(self._nuid + 1, dtype=np.int64)

        # Set starts
        self._user_offsets[unique_users] = start_indices
        # Set ends (start of next user)
        self._user_offsets[unique_users + 1] = np.roll(start_indices, -1)
        self._user_offsets[-1] = len(self._flat_items)

        # Fill gaps for users with no interactions (propagate previous offset)
        # This ensures user_offsets[u] == user_offsets[u+1] for empty users
        for i in range(1, len(self._user_offsets)):
            if self._user_offsets[i] == 0 and self._user_offsets[i - 1] > 0:
                self._user_offsets[i] = self._user_offsets[i - 1]

    def get_user_history_sequences(
        self, user_ids: List[int], max_seq_len: int
    ) -> Tuple[Tensor, Tensor]:
        """Retrieves padded historical sequences for inference/evaluation."""
        if not self._cached_user_histories:
            # Build dict cache on demand
            # Using split on the flat array is faster than iterating DF
            starts = self._user_offsets[:-1]
            ends = self._user_offsets[1:]
            # Only for users that actually exist in data
            valid_u = np.where(ends > starts)[0]
            self._cached_user_histories = {
                int(u): self._flat_items[starts[u] : ends[u]].tolist() for u in valid_u
            }

        seqs, lens = [], []
        for uid in user_ids:
            hist = self._cached_user_histories.get(uid, [])
            recent = hist[-max_seq_len:]
            seqs.append(torch.tensor(recent, dtype=torch.long))
            lens.append(len(recent))

        sequences = pad_sequence(seqs, batch_first=True, padding_value=self._niid)

        # A batch in which no user has any history, which a cold start protocol
        # produces as soon as the held out users land together, pads to a width
        # of zero and leaves a sequential model with no position to read at all.
        # One column of padding is the shortest sequence they can be asked
        # about, and is what a single cold user in a warm batch already gets.
        if sequences.shape[1] == 0:
            sequences = sequences.new_full((len(user_ids), 1), self._niid)

        return sequences, torch.tensor(lens, dtype=torch.long)

    def _require_valid_targets(self) -> None:
        """Work out which positions can be predicted, and insist there are some.

        A position can only be a target if something precedes it, so the first
        interaction of every user is excluded. The result is worked out once and
        kept, because both sequential loaders ask for the same thing.

        Raises:
            ValueError: If no position in the data has a predecessor.
        """
        if self._valid_sample_indices is None:
            positions = np.arange(len(self._flat_items))
            usable = np.ones(len(self._flat_items), dtype=bool)

            # The first item of any user cannot be a target: it has no history.
            starts = self._user_offsets[:-1]
            usable[starts[starts < len(self._flat_items)]] = False

            self._valid_sample_indices = positions[usable]

        if len(self._valid_sample_indices) == 0:
            raise ValueError(
                "No valid sequences found (min 2 interactions per user needed)."
            )

    def get_sequential_dataloader(
        self,
        max_seq_len: int,
        neg_samples: int = 0,
        include_user_id: bool = False,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ) -> DataLoader:
        """Standard SASRec/RNN style dataloader (History -> Next Item)."""

        self._require_valid_targets()

        dataset = SequentialDataset(
            flat_items=self._flat_items,
            flat_users=self._flat_users,
            user_offsets=self._user_offsets,
            valid_target_indices=self._valid_sample_indices,
            sparse_matrix=self._inter_sparse,
            max_seq_len=max_seq_len,
            neg_samples=neg_samples,
            niid=self._niid,
            include_user_id=include_user_id,
            seed=seed,
        )

        return seeded_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            **kwargs,
        )

    def get_same_target_sequential_dataloader(
        self,
        max_seq_len: int,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        low_memory: bool = False,
        **kwargs: Any,
    ) -> DataLoader:
        """Sequential dataloader that also samples a same-target positive sequence."""

        self._require_valid_targets()

        dataset = SameTargetSequentialDataset(
            flat_items=self._flat_items,
            flat_users=self._flat_users,
            user_offsets=self._user_offsets,
            valid_target_indices=self._valid_sample_indices,
            max_seq_len=max_seq_len,
            niid=self._niid,
            seed=seed,
        )

        return seeded_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            **kwargs,
        )

    def get_sliding_window_dataloader(
        self,
        max_seq_len: int,
        neg_samples: int,
        stride: int = 1,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ) -> DataLoader:
        """Sequence-to-Sequence dataloader (Sliding Windows)."""

        # 1. Calculate Windows
        user_lens = np.diff(self._user_offsets)
        valid_users = np.where(user_lens >= 2)[0]

        if len(valid_users) == 0:
            raise ValueError("No valid sliding windows found.")

        valid_lens = user_lens[valid_users]
        valid_starts = self._user_offsets[valid_users]

        # Number of windows per user
        num_windows = (
            np.floor((np.maximum(valid_lens - max_seq_len, 0) / stride)).astype(int) + 1
        )
        total_samples = np.sum(num_windows)

        # 2. Map Dataset Index -> (User, Start_Index)
        # Repeat user IDs
        window_user_ids = np.repeat(valid_users, num_windows)

        # Calculate start indices
        # Cumulative count of windows to find offsets
        cum_windows = np.zeros(len(valid_users) + 1, dtype=int)
        cum_windows[1:] = np.cumsum(num_windows)

        indices = np.arange(total_samples)
        # Find which user block each index belongs to
        user_block_indices = np.searchsorted(cum_windows, indices, side="right") - 1

        # Local index within the user's windows (0, 1, 2...)
        local_window_idx = indices - cum_windows[user_block_indices]

        # Map back to flat array index
        # Start = UserStart + (WindowIndex * Stride)
        # Note: We use user_block_indices to index valid_starts because they align with valid_users
        window_starts_flat = valid_starts[user_block_indices] + (
            local_window_idx * stride
        )

        dataset = SlidingWindowDataset(
            flat_items=self._flat_items,
            window_starts=window_starts_flat.astype(np.int64),
            window_users=window_user_ids.astype(np.int64),
            sparse_matrix=self._inter_sparse,
            max_seq_len=max_seq_len,
            neg_samples=neg_samples,
            niid=self._niid,
            seed=seed,
        )

        return seeded_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            **kwargs,
        )

    def get_cloze_mask_dataloader(
        self,
        max_seq_len: int,
        mask_prob: float,
        mask_token_id: int,
        neg_samples: int,
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ) -> DataLoader:
        """BERT4Rec style dataloader (Masked Language Modeling)."""

        user_lens = np.diff(self._user_offsets)
        valid_users = np.where(user_lens >= 2)[0]

        if len(valid_users) == 0:
            raise ValueError("No valid users with >= 2 interactions found.")

        # Use the last available window for Cloze task
        user_starts = self._user_offsets[valid_users]
        user_ends = self._user_offsets[valid_users + 1]

        # Start is at least (End - MaxLen)
        window_starts = np.maximum(user_starts, user_ends - max_seq_len)

        dataset = ClozeDataset(
            flat_items=self._flat_items,
            window_starts=window_starts.astype(np.int64),
            window_ends=user_ends.astype(np.int64),
            window_users=valid_users.astype(np.int64),
            sparse_matrix=self._inter_sparse,
            max_seq_len=max_seq_len,
            mask_prob=mask_prob,
            mask_token_id=mask_token_id,
            neg_samples=neg_samples,
            niid=self._niid,
            seed=seed,
        )

        return seeded_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            **kwargs,
        )
