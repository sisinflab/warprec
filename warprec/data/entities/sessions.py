from typing import Optional, List, Any, Tuple, Dict

import torch
import numpy as np
import narwhals as nw
from narwhals.dataframe import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix

from warprec.data.entities.train_structures import (
    SequentialDataset,
    SlidingWindowDataset,
    ClozeDataset,
)


def seed_worker(worker_id):
    """Ensures reproducibility in DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


class Sessions:
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

        native_ns = nw.get_native_namespace(self._inter_df)

        # Create mapping frames
        umap_df = nw.from_dict(
            {
                self.user_label: list(self._umap.keys()),
                "__uidx__": list(self._umap.values()),
            },
            native_namespace=native_ns,
        )
        imap_df = nw.from_dict(
            {
                self.item_label: list(self._imap.keys()),
                "__iidx__": list(self._imap.values()),
            },
            native_namespace=native_ns,
        )

        # Join and Map
        mapped_df = (
            self._inter_df.join(umap_df, on=self.user_label, how="inner")
            .join(imap_df, on=self.item_label, how="inner")
            .select(
                [
                    nw.col("__uidx__").alias(self.user_label).cast(nw.Int64),
                    nw.col("__iidx__").alias(self.item_label).cast(nw.Int64),
                    # Keep timestamp if exists
                    *(
                        [nw.col(self.timestamp_label)]
                        if self.timestamp_label in self._inter_df.columns
                        else []
                    ),
                    # Keep context if exists
                    *(
                        [
                            nw.col(c).cast(nw.Int64)
                            for c in self.context_labels
                            if c in self._inter_df.columns
                        ]
                    ),
                ]
            )
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
                u: self._flat_items[starts[u] : ends[u]].tolist() for u in valid_u
            }

        seqs, lens = [], []
        for uid in user_ids:
            hist = self._cached_user_histories.get(uid, [])
            recent = hist[-max_seq_len:]
            seqs.append(torch.tensor(recent, dtype=torch.long))
            lens.append(len(recent))

        return (
            pad_sequence(seqs, batch_first=True, padding_value=self._niid),
            torch.tensor(lens, dtype=torch.long),
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

        # Identify valid targets (items that have at least 1 predecessor)
        if self._valid_sample_indices is None:
            all_indices = np.arange(len(self._flat_items))
            valid_mask = np.ones(len(self._flat_items), dtype=bool)

            # The first item of any user cannot be a target (no history)
            user_starts = self._user_offsets[:-1]
            # Filter only starts that are within bounds (active users)
            active_starts = user_starts[user_starts < len(self._flat_items)]

            valid_mask[active_starts] = False
            self._valid_sample_indices = all_indices[valid_mask]

        if len(self._valid_sample_indices) == 0:
            raise ValueError(
                "No valid sequences found (min 2 interactions per user needed)."
            )

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
        )

        g = torch.Generator()
        g.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
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
        )

        g = torch.Generator()
        g.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
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

        g = torch.Generator()
        g.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            worker_init_fn=seed_worker,
            generator=g,
            **kwargs,
        )
