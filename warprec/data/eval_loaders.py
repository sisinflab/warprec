from typing import Tuple, List, Dict, Any

import torch
import numpy as np
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
from scipy.sparse import csr_matrix


class EvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, dense_ground_truth)
    """

    def __init__(
        self,
        eval_interactions: csr_matrix,
    ):
        self.num_users = eval_interactions.shape[0]
        self.eval_interactions = eval_interactions

    def __len__(self) -> int:
        return self.num_users

    def __getitem__(self, idx: int) -> Tuple[int, Tensor]:
        eval_row = self.eval_interactions.getrow(idx)
        ground_truth = torch.from_numpy(eval_row.toarray()).to(torch.float32).squeeze(0)

        return idx, ground_truth


class ContextualEvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, target_item_idx, context_vector)
    """

    def __init__(
        self,
        eval_data: DataFrame,
        user_id_label: str,
        item_id_label: str,
        context_labels: List[str],
    ):
        # Pre-convert DataFrames to torch tensor to reduce overhead
        self.user_indices = torch.from_numpy(
            eval_data[user_id_label].values.astype(np.int64)
        )
        self.item_indices = torch.from_numpy(
            eval_data[item_id_label].values.astype(np.int64)
        )
        self.context_features = torch.from_numpy(
            eval_data[context_labels].values.astype(np.int64)
        )

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self.user_indices[idx],
            self.item_indices[idx],
            self.context_features[idx],
        )


class SequentialEvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, target_item_idx, sequence_tensor)
    """

    def __init__(
        self,
        train_df: DataFrame,
        eval_df: DataFrame,
        user_id_label: str,
        item_id_label: str,
        max_seq_len: int,
    ):
        self.max_seq_len = max_seq_len

        # Internal storage for samples: List of (user_id, target_index_in_full_history)
        self.samples: List[Tuple[int, int]] = []

        # Internal storage for full history: Dict[user_id, List[item_id]]
        self.user_history: Dict[int, List[int]] = {}

        self._preprocess_data(train_df, eval_df, user_id_label, item_id_label)

    def _preprocess_data(
        self, train_df: DataFrame, eval_df: DataFrame, user_col: str, item_col: str
    ):
        # Grouping by user to get lists of items
        # NOTE: Assumes DataFrames are already sorted by time/interaction order
        train_groups = train_df.groupby(user_col)[item_col].apply(list)
        eval_groups = eval_df.groupby(user_col)[item_col].apply(list)

        # Identify all unique users involved in evaluation
        eval_users = eval_groups.index.unique()

        for user_id in eval_users:
            # Get histories (default to empty list if user not in train)
            train_seq = train_groups.get(user_id, [])
            eval_seq = eval_groups.get(user_id, [])

            if not eval_seq:
                continue

            # Combine to create full timeline
            full_history = train_seq + eval_seq
            self.user_history[user_id] = full_history

            # The index where Eval data starts in the full history
            start_eval_idx = len(train_seq)

            # Create a sample for EACH item in the evaluation set
            # The target is at index 'i' in full_history
            # The sequence is full_history[:i]
            for i in range(len(eval_seq)):
                target_idx_in_history = start_eval_idx + i
                self.samples.append((user_id, target_idx_in_history))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[int, Tensor, Tensor]:
        user_id, target_idx = self.samples[idx]
        full_history = self.user_history[user_id]

        # Target Item
        target_item = full_history[target_idx]

        # Input Sequence: Everything before target
        # Slicing logic: take up to target_idx, then take last max_seq_len
        seq_list = full_history[:target_idx]
        seq_list = seq_list[-self.max_seq_len :]

        seq_tensor = torch.tensor(seq_list, dtype=torch.long)
        target_tensor = torch.tensor(target_item, dtype=torch.long)

        return user_id, target_tensor, seq_tensor


class SampledEvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, pos_item, neg_items_vector)
    """

    def __init__(
        self,
        train_interactions: csr_matrix,
        eval_interactions: csr_matrix,
        num_negatives: int = 99,
        seed: int = 42,
    ):
        super().__init__()
        self.num_users, self.num_items = train_interactions.shape

        # Pre-calculate all positives (Train + Eval) per user,
        # we use a list of arrays for fast access
        self.all_positives = []
        for u in range(self.num_users):
            train_indices = train_interactions.indices[
                train_interactions.indptr[u] : train_interactions.indptr[u + 1]
            ]
            eval_indices = eval_interactions.indices[
                eval_interactions.indptr[u] : eval_interactions.indptr[u + 1]
            ]
            self.all_positives.append(np.union1d(train_indices, eval_indices))

        # Identify users who actually have evaluation data
        # (We only want to iterate over these)
        self.users_with_eval = [
            u
            for u in range(self.num_users)
            if eval_interactions.indptr[u + 1] - eval_interactions.indptr[u] > 0
        ]

        self.positive_items_list = []
        self.negative_items_list = []

        np.random.seed(seed)

        for u in self.users_with_eval:
            # Store positives
            eval_pos = eval_interactions.indices[
                eval_interactions.indptr[u] : eval_interactions.indptr[u + 1]
            ]
            self.positive_items_list.append(torch.tensor(eval_pos, dtype=torch.long))

            # Compute seen items
            seen_items = self.all_positives[u]
            n_seen = len(seen_items)

            # If user has seen almost everything, return empty or partial
            if self.num_items - n_seen <= 0:
                self.negative_items_list.append(torch.tensor([], dtype=torch.long))
                continue

            # Sample one time 2x the number of negatives
            # NOTE: In most cases this will skip the while loop
            num_to_generate = num_negatives * 2
            candidates = np.random.randint(0, self.num_items, size=num_to_generate)

            # Fast filtering using numpy boolean masking
            # Note: For very large item sets, np.isin can be slow.
            # If num_items > 1M, consider Bloom Filters or just the 'while' loop.
            mask = np.isin(candidates, seen_items, invert=True)
            valid_negatives = candidates[mask]

            # Remove duplicates in candidates if necessary
            valid_negatives = np.unique(valid_negatives)

            # If we don't have enough, fallback to a loop
            if len(valid_negatives) < num_negatives:
                final_negs = list(valid_negatives)
                while len(final_negs) < num_negatives:
                    cand = np.random.randint(0, self.num_items)
                    if cand not in seen_items and cand not in final_negs:
                        final_negs.append(cand)
                valid_negatives = np.array(final_negs)

            # Take exactly num_negatives
            self.negative_items_list.append(
                torch.tensor(valid_negatives[:num_negatives], dtype=torch.long)
            )

    def __len__(self) -> int:
        return len(self.users_with_eval)

    def __getitem__(self, idx: int) -> Tuple[int, Tensor, Tensor]:
        user_idx = self.users_with_eval[idx]
        return (
            user_idx,
            self.positive_items_list[idx],
            self.negative_items_list[idx],
        )


class SampledContextualEvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, pos_item, neg_items_vector, context_vector)
    """

    def __init__(
        self,
        train_interactions: csr_matrix,
        eval_data: DataFrame,
        user_id_label: str,
        item_id_label: str,
        context_labels: List[str],
        num_items: int,
        num_negatives: int = 99,
        seed: int = 42,
    ):
        # pylint: disable = too-many-nested-blocks
        self.num_negatives = num_negatives
        self.num_items = num_items

        # Pre-convert DataFrames to torch tensor to reduce overhead
        self.user_indices = torch.from_numpy(
            eval_data[user_id_label].values.astype(np.int64)
        )
        self.pos_item_indices = torch.from_numpy(
            eval_data[item_id_label].values.astype(np.int64)
        )
        self.context_features = torch.from_numpy(
            eval_data[context_labels].values.astype(np.int64)
        )

        n_train_users = train_interactions.shape[0]

        self.negatives_list: list[Tensor] = []
        np.random.seed(seed)

        for idx, user_idx_tensor in enumerate(self.user_indices):
            u = int(user_idx_tensor.item())
            target_item = int(self.pos_item_indices[idx].item())

            # Retrieve training history
            if u < n_train_users:
                train_items = train_interactions.indices[
                    train_interactions.indptr[u] : train_interactions.indptr[u + 1]
                ]
            else:
                train_items = np.array([], dtype=np.int64)

            # Compute seen items
            seen_items = np.append(train_items, target_item)

            # Generate 2x candidates to avoid loops in most cases
            num_to_generate = self.num_negatives * 2
            candidates = np.random.randint(0, self.num_items, size=num_to_generate)

            # Filter out seen items using optimized numpy boolean masking
            mask = np.isin(candidates, seen_items, invert=True)
            valid_negatives = candidates[mask]

            # Remove duplicates
            valid_negatives = np.unique(valid_negatives)

            # If we don't have enough, fallback to a loop
            if len(valid_negatives) < self.num_negatives:
                final_negs = list(valid_negatives)
                while len(final_negs) < self.num_negatives:
                    cand = np.random.randint(0, self.num_items)
                    if cand not in final_negs:
                        if cand != target_item:
                            if cand not in train_items:
                                final_negs.append(cand)  # type: ignore[arg-type]
                valid_negatives = np.array(final_negs)

            # Take exactly num_negatives
            self.negatives_list.append(
                torch.tensor(valid_negatives[: self.num_negatives], dtype=torch.long)
            )

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.user_indices[idx],
            self.pos_item_indices[idx],
            self.negatives_list[idx],
            self.context_features[idx],
        )


class SampledSequentialEvaluationDataset(TorchDataset):
    """
    Yields: (user_idx, pos_item, neg_items_vector, sequence_tensor)
    """

    def __init__(
        self,
        train_df: DataFrame,
        eval_df: DataFrame,
        user_id_label: str,
        item_id_label: str,
        max_seq_len: int,
        num_items: int,
        num_negatives: int = 99,
        seed: int = 42,
    ):
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.num_negatives = num_negatives

        # Internal storage for samples: List of (user_id, target_index_in_full_history)
        self.samples: List[Tuple[int, int]] = []

        # Internal storage for full history: Dict[user_id, List[item_id]]
        self.user_history: Dict[int, List[int]] = {}

        # Cache for seen items (as numpy arrays) for fast negative sampling
        self.seen_items_cache: Dict[int, np.ndarray] = {}

        # Initialize random seed
        np.random.seed(seed)

        self._preprocess_data(train_df, eval_df, user_id_label, item_id_label)

    def _preprocess_data(
        self, train_df: DataFrame, eval_df: DataFrame, user_col: str, item_col: str
    ):
        # Grouping by user to get lists of items
        train_groups = train_df.groupby(user_col)[item_col].apply(list)
        eval_groups = eval_df.groupby(user_col)[item_col].apply(list)

        # Identify all unique users involved in evaluation
        eval_users = eval_groups.index.unique()

        for user_id in eval_users:
            # Get histories
            train_seq = train_groups.get(user_id, [])
            eval_seq = eval_groups.get(user_id, [])

            if not eval_seq:
                continue

            # Combine to create full timeline
            full_history = train_seq + eval_seq
            self.user_history[user_id] = full_history

            # Store seen items as numpy array for fast 'np.isin' checks later
            self.seen_items_cache[user_id] = np.array(full_history)

            # The index where Eval data starts in the full history
            start_eval_idx = len(train_seq)

            # Create a sample for EACH item in the evaluation set
            for i in range(len(eval_seq)):
                target_idx_in_history = start_eval_idx + i
                self.samples.append((user_id, target_idx_in_history))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[int, Tensor, Tensor, Tensor]:
        user_id, target_idx = self.samples[idx]
        full_history = self.user_history[user_id]

        # Target Item (Positive)
        target_item = full_history[target_idx]

        # Input Sequence
        seq_list = full_history[:target_idx]
        seq_list = seq_list[-self.max_seq_len :]
        seq_tensor = torch.tensor(seq_list, dtype=torch.long)

        # Negative Sampling
        seen_items = self.seen_items_cache[user_id]

        # Check if sampling is possible
        if self.num_items - len(seen_items) <= 0:
            # Fallback: return empty or random if user saw everything (edge case)
            negatives = torch.randint(
                0, self.num_items, (self.num_negatives,), dtype=torch.long
            )
        else:
            # Generate 2x candidates to avoid loops in most cases
            num_to_generate = self.num_negatives * 2
            candidates = np.random.randint(0, self.num_items, size=num_to_generate)

            # Filter out seen items
            mask = np.isin(candidates, seen_items, invert=True)
            valid_negatives = candidates[mask]
            valid_negatives = np.unique(valid_negatives)

            # Fallback loop if not enough negatives
            if len(valid_negatives) < self.num_negatives:
                final_negs: List[Any] = list(valid_negatives)
                while len(final_negs) < self.num_negatives:
                    cand = np.random.randint(0, self.num_items)
                    if cand not in seen_items and cand not in final_negs:
                        final_negs.append(cand)
                valid_negatives = np.array(final_negs)

            negatives = torch.tensor(
                valid_negatives[: self.num_negatives], dtype=torch.long
            )

        return (
            user_id,
            torch.tensor(target_item, dtype=torch.long),
            negatives,
            seq_tensor,
        )


class EvaluationDataLoader(DataLoader):
    """
    Output Batch: (user_indices, ground_truths)
    """

    def __init__(
        self,
        eval_interactions: csr_matrix,
        batch_size: int = 1024,
        **kwargs,
    ):
        dataset = EvaluationDataset(
            eval_interactions=eval_interactions,
        )
        super().__init__(dataset, batch_size=batch_size, shuffle=False, **kwargs)


class ContextualEvaluationDataLoader(DataLoader):
    """
    Output Batch: (user_indices, target_items, contexts)
    """

    def __init__(
        self,
        eval_data: DataFrame,
        user_id_label: str,
        item_id_label: str,
        context_labels: List[str],
        batch_size: int = 1024,
        **kwargs,
    ):
        dataset = ContextualEvaluationDataset(
            eval_data=eval_data,
            user_id_label=user_id_label,
            item_id_label=item_id_label,
            context_labels=context_labels,
        )
        super().__init__(dataset, batch_size=batch_size, shuffle=False, **kwargs)


class SequentialEvaluationDataLoader(DataLoader):
    """
    Output Batch: (user_indices, target_items, padded_sequences)
    """

    def __init__(
        self,
        train_data: DataFrame,
        eval_data: DataFrame,
        user_id_label: str,
        item_id_label: str,
        max_seq_len: int,
        batch_size: int = 1024,
        **kwargs,
    ):
        self.num_items = train_data[item_id_label].nunique()

        dataset = SequentialEvaluationDataset(
            train_df=train_data,
            eval_df=eval_data,
            user_id_label=user_id_label,
            item_id_label=item_id_label,
            max_seq_len=max_seq_len,
        )

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            **kwargs,
        )

    def _collate_fn(
        self,
        batch: List[Tuple[int, Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        user_indices, target_tensors, seq_tensors = zip(*batch)

        user_indices_tensor = torch.tensor(list(user_indices), dtype=torch.long)
        target_items_tensor = torch.stack(target_tensors)

        seqs_padded = pad_sequence(
            seq_tensors,  # type: ignore[arg-type]
            batch_first=True,
            padding_value=self.num_items,
        )

        return user_indices_tensor, target_items_tensor, seqs_padded


class SampledEvaluationDataLoader(DataLoader):
    """
    Output Batch: (user_indices, pos_items, neg_items)
    """

    def __init__(
        self,
        train_interactions: csr_matrix,
        eval_interactions: csr_matrix,
        num_negatives: int = 99,
        seed: int = 42,
        batch_size: int = 1024,
        **kwargs,
    ):
        self.num_items = train_interactions.shape[1]

        dataset = SampledEvaluationDataset(
            train_interactions=train_interactions,
            eval_interactions=eval_interactions,
            num_negatives=num_negatives,
            seed=seed,
        )

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            **kwargs,
        )

    def _collate_fn(
        self,
        batch: List[Tuple[int, Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        user_indices, positive_tensors, negative_tensors = zip(*batch)

        user_indices_tensor = torch.tensor(list(user_indices), dtype=torch.long)

        # Stack sequences and pad them ensure same length
        positives_padded = pad_sequence(
            positive_tensors,  # type: ignore[arg-type]
            batch_first=True,
            padding_value=self.num_items,
        )
        negatives_padded = pad_sequence(
            negative_tensors,  # type: ignore[arg-type]
            batch_first=True,
            padding_value=self.num_items,
        )

        return user_indices_tensor, positives_padded, negatives_padded


class SampledContextualEvaluationDataLoader(DataLoader):
    """
    Output Batch: (user_indices, pos_items, neg_items, contexts)
    """

    def __init__(
        self,
        train_interactions: csr_matrix,
        eval_data: DataFrame,
        user_id_label: str,
        item_id_label: str,
        context_labels: List[str],
        num_items: int,
        num_negatives: int = 99,
        seed: int = 42,
        batch_size: int = 1024,
        **kwargs,
    ):
        dataset = SampledContextualEvaluationDataset(
            train_interactions=train_interactions,
            eval_data=eval_data,
            user_id_label=user_id_label,
            item_id_label=item_id_label,
            context_labels=context_labels,
            num_items=num_items,
            num_negatives=num_negatives,
            seed=seed,
        )

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            **kwargs,
        )

    def _collate_fn(
        self,
        batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        user_indices, pos_items, neg_items, context_features = zip(*batch)

        tensor_user_indices = torch.stack(user_indices)
        tensor_pos_items = torch.stack(pos_items).unsqueeze(1)
        tensor_neg_items = torch.stack(neg_items)
        tensor_context_features = torch.stack(context_features)

        return (
            tensor_user_indices,
            tensor_pos_items,
            tensor_neg_items,
            tensor_context_features,
        )


class SampledSequentialEvaluationDataLoader(DataLoader):
    """
    Output Batch: (user_indices, pos_items, neg_items, padded_sequences)
    """

    def __init__(
        self,
        train_data: DataFrame,
        eval_data: DataFrame,
        user_id_label: str,
        item_id_label: str,
        max_seq_len: int,
        num_negatives: int = 99,
        seed: int = 42,
        batch_size: int = 1024,
        **kwargs,
    ):
        # Infer num_items from data if possible, or pass it explicitly if needed.
        # Here we assume item_ids are contiguous 0..N-1, so max_id + 1 is num_items.
        # Ideally, this should be passed as an argument to be safe.
        max_train = train_data[item_id_label].max()
        max_eval = eval_data[item_id_label].max()
        self.num_items = max(max_train, max_eval) + 1

        dataset = SampledSequentialEvaluationDataset(
            train_df=train_data,
            eval_df=eval_data,
            user_id_label=user_id_label,
            item_id_label=item_id_label,
            max_seq_len=max_seq_len,
            num_items=self.num_items,
            num_negatives=num_negatives,
            seed=seed,
        )

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            **kwargs,
        )

    def _collate_fn(
        self,
        batch: List[Tuple[int, Tensor, Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        user_indices, pos_tensors, neg_tensors, seq_tensors = zip(*batch)

        user_indices_tensor = torch.tensor(list(user_indices), dtype=torch.long)

        # Positives: Stack and unsqueeze to get (Batch, 1)
        # This matches the shape (Batch, Num_Negatives) for easy concatenation later
        pos_items_tensor = torch.stack(pos_tensors).unsqueeze(1)

        # Negatives: Stack to get (Batch, Num_Negatives)
        neg_items_tensor = torch.stack(neg_tensors)

        # Sequences: Pad to (Batch, Max_Len_In_Batch)
        # Using num_items as padding value to match your previous Sequential loader
        seqs_padded = pad_sequence(
            seq_tensors,  # type: ignore[arg-type]
            batch_first=True,
            padding_value=self.num_items,
        )

        return user_indices_tensor, pos_items_tensor, neg_items_tensor, seqs_padded
